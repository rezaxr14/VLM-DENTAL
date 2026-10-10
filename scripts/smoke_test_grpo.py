#!/usr/bin/env python3
"""End-to-end smoke test of ``scripts/run_grpo.py`` (and optionally ``scripts/evaluate_models.py``) with a tiny random
Qwen3.5 (no downloads, no real weights).

It builds the tiny model of ``smoke_test_sft.py``, saves an untrained LoRA adapter as the "SFT reference", creates a
synthetic two-image dataset, and runs the real GRPO launcher (rollouts, rewards, advantages, policy update, checkpoint
save) in-process on CPU. Generation is capped to a few tokens and the reward is replaced by a deterministic stand-in so
that advantages are non-zero; this checks plumbing, gradient flow and library compatibility, not learning, memory or speed.

    python scripts/smoke_test_grpo.py --traces data/traces/train_cot_traces.jsonl --track no_tools -- --canonical-resize
    # additionally run evaluation of the SFT adapter and the GRPO result (4-bit, as on a 24 GB GPU)
    python scripts/smoke_test_grpo.py --traces ... --track with_tools --eval-precision qlora -- --canonical-resize
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

MAX_NEW_TOKENS = 24


def synthetic_dataset(data_dir: Path, source: str | None = "tufts"):
    import numpy as np
    import pandas as pd
    from PIL import Image

    rng = np.random.default_rng(0)
    rows, annots = [], []
    for i in (1, 2):
        p = data_dir / f"img_{i}.png"
        Image.fromarray(rng.normal(110, 40, (840, 1615)).clip(0, 255).astype("uint8")).convert("RGB").save(p)
        rows.append({"id": i, "local_path": str(p)})
        row = {"image_id": i, "category_id_1": 3, "category_id_2": 6, "category_id_3": 1, "bbox": [600.0, 300.0, 120.0, 160.0]}
        if source:
            row["source_dataset"] = source
        annots.append(row)
    categories = pd.DataFrame({"id": [1, 2], "name": ["Caries", "Periapical Lesion"]})
    return pd.DataFrame(rows), pd.DataFrame(annots), categories


def main() -> int:
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="Any trace JSONL (its text trains the tiny tokenizer)")
    ap.add_argument("--track", default="no_tools", choices=["no_tools", "with_tools"])
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--eval-precision", default=None, choices=["bf16", "qlora"],
                    help="Also run evaluate_models.py on the SFT adapter and the GRPO result with this --precision")
    args = ap.parse_args(argv)

    import torch
    from safetensors.torch import load_file
    from transformers import GenerationMixin, Qwen3_5ForConditionalGeneration
    import peft

    from dental_agent.model.backbone import lora_target_modules
    from scripts.smoke_test_sft import build_tiny_model

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="grpo_smoke_"))
    model_dir, data_dir, adapter_dir, out_dir = work / "model", work / "data", work / "adapter", work / "out"
    for d in (model_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in open(args.traces, encoding="utf-8") if line.strip()]
    print(f"[SMOKE-GRPO] building tiny Qwen3.5 in {model_dir} ...")
    build_tiny_model(records, model_dir, four_bit_friendly=args.eval_precision == "qlora")

    base = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32)
    cfg = peft.LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM",
                          target_modules=lora_target_modules(base, linear_attention=True, vision_projector=True))
    peft.get_peft_model(base, cfg).save_pretrained(adapter_dir)
    del base

    images_df, annots_df, categories_df = synthetic_dataset(data_dir)

    import scripts.run_grpo as run_grpo
    import dental_agent.training.grpo as grpo

    run_grpo.load_tufts_dataset = lambda *_a, **_k: (images_df, annots_df, categories_df)

    original_generate = GenerationMixin.generate

    def capped_generate(self, *a, **k):
        k["max_new_tokens"] = min(int(k.get("max_new_tokens") or MAX_NEW_TOKENS), MAX_NEW_TOKENS)
        k.pop("max_length", None)
        return original_generate(self, *a, **k)

    GenerationMixin.generate = capped_generate

    # Stand-in reward: random tiny-model output earns the same reward every time, which would zero every advantage.
    grpo.combine_reward = lambda traj, gt, max_tool_calls=0: (
        (len(json.dumps(traj, default=str)) % 17) / 17.0, {})

    # The step log defaults to ./data/grpo_training_log.jsonl, the real training log; keep smoke rows out of it.
    real_log_step = grpo.log_grpo_step
    grpo.log_grpo_step = lambda stats, log_path=None, extra=None: real_log_step(
        stats, log_path=work / "grpo_training_log.jsonl", extra=extra)

    sys.argv = [
        "run_grpo.py", "--track", args.track, "--sft-stage", "dentex_alone", "--sft-model-dir", str(adapter_dir),
        "--model-id", str(model_dir), "--dataset", "tufts", "--group-size", "2", "--epochs", "1",
        "--output-dir", str(out_dir), "--hf-repo", "", "--push-every-steps", "1000", "--lr", "1e-3", *passthrough,
    ]
    os.environ["HF_HUB_OFFLINE"] = "1"
    run_grpo.main()

    finals = glob.glob(str(out_dir / "**" / "adapter_model.safetensors"), recursive=True)
    moved = False
    for path in finals:
        moved |= any("lora_B" in k and t.abs().sum() > 0 for k, t in load_file(path).items())
    ok = bool(finals) and moved

    evaluated = None
    if ok and args.eval_precision:
        import scripts.evaluate_models as ev

        eval_images, eval_annots, eval_cats = synthetic_dataset(data_dir, source=None)
        ev.load_dentex_dataset = lambda *_a, **_k: (eval_images, eval_annots, eval_cats)
        grpo_final = next(Path(p).parent.parent for p in finals if p.endswith("grpo_policy/adapter_model.safetensors") and "final" in p)
        evaluated = []
        for condition, adapter in ((f"sft_{args.track}", adapter_dir), (f"grpo_{args.track}", grpo_final)):
            sys.argv = ["evaluate_models.py", "--condition", condition, "--adapter-path", str(adapter),
                        "--model-id", str(model_dir), "--dataset", "dentex", "--split", "test", "--limit", "2",
                        "--output-dir", str(work / "eval"), "--precision", args.eval_precision,
                        "--max-turns", "3", "--max-tool-calls", "3"]
            ev.main()
            evaluated.append(condition)
        ok = any((work / "eval").rglob("*.json*"))

    extra = f", evaluated: {evaluated}" if evaluated is not None else ""
    print(f"[SMOKE-GRPO] {'PASSED' if ok else 'FAILED'} (checkpoints: {len(finals)}, LoRA weights updated: {moved}{extra})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
