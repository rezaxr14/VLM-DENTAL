#!/usr/bin/env python3
"""End-to-end smoke test of ``scripts/train_sft.py`` with a tiny random Qwen3.5 (no downloads, no real weights).

What it exercises, with YOUR installed ``transformers`` / ``peft`` / ``torch``: the real trace files, the real tool
pipeline, canonical views, the collator, LoRA wiring, gradient checkpointing, the training/eval loop and checkpoint saving.
It does NOT measure memory or speed (the model has ~2M parameters) -- it catches crashes and version incompatibilities
before a long run on a GPU/TPU.

Examples (every argument after ``--`` is passed to train_sft.py verbatim):

    # the configuration planned for a single 24 GB GPU
    python scripts/smoke_test_sft.py --traces data/traces/train_cot_traces.jsonl -- \
        --canonical-resize --max-seq-len 8192
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
    "{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for it in m['content'] %}"
    "{% if it['type']=='image' %}<|vision_start|><|image_pad|><|vision_end|>{% elif it['type']=='text' %}{{ it['text'] }}{% endif %}"
    "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build_tiny_model(traces: list[dict], out_dir: Path) -> None:
    """Tiny hybrid (linear + full attention) Qwen3.5 with a real ViT, plus a BPE tokenizer trained on the traces."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForConditionalGeneration, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    texts = []
    for rec in traces:
        for m in rec["messages"]:
            c = m["content"]
            texts.append(c if isinstance(c, str) else " ".join((i.get("text") or "") for i in c))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        texts,
        trainers.BpeTrainer(vocab_size=12000, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=SPECIAL[1:], chat_template=TEMPLATE,
    )
    ids = {t: fast.convert_tokens_to_ids(t) for t in SPECIAL}
    image_processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, temporal_patch_size=2, size={"shortest_edge": 65536, "longest_edge": 16777216}
    )
    Qwen3VLProcessor(
        image_processor=image_processor, tokenizer=fast, video_processor=Qwen3VLVideoProcessor(), chat_template=TEMPLATE
    ).save_pretrained(out_dir)

    text_cfg = Qwen3_5TextConfig(
        vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=32768,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    vision_cfg = Qwen3_5VisionConfig(
        depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64, patch_size=16,
        spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304,
    )
    cfg = Qwen3_5Config(
        text_config=text_cfg, vision_config=vision_cfg, image_token_id=ids["<|image_pad|>"],
        video_token_id=ids["<|video_pad|>"], vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"], tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(out_dir)


def select_traces(records: list[dict], n: int, max_images: int) -> list[dict]:
    """Shortest traces (by characters) that use at most ``max_images`` images; keeps the smoke run small."""
    from scripts.census_vision_slots import count_families

    pool = [r for r in records if sum(count_families(r).values()) <= max_images]
    pool.sort(key=lambda r: sum(len(json.dumps(m["content"])) for m in r["messages"]))
    return pool[:n]


def write_fake_images(records: list[dict], data_dir: Path) -> list[dict]:
    """Noise radiographs at each dataset's real resolution; tools (crop, contrast, denoise...) run on them for real."""
    import numpy as np
    from PIL import Image

    sizes = {"dentex": (2872, 1504), "tufts": (1615, 840)}
    rng = np.random.default_rng(0)
    paths: dict[str, str] = {}
    for name, (w, h) in sizes.items():
        p = data_dir / f"{name}_fake.png"
        Image.fromarray(rng.normal(110, 40, (h, w)).clip(0, 255).astype("uint8")).convert("RGB").save(p)
        paths[name] = str(p)
    out = []
    for r in records:
        r = dict(r)
        r["image_path"] = paths.get(r.get("dataset", "dentex"), paths["dentex"])
        out.append(r)
    return out


def main() -> int:
    argv = sys.argv[1:]
    passthrough: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="With-tools trace JSONL (e.g. data/traces/train_cot_traces.jsonl)")
    ap.add_argument("--num-traces", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=5, help="Skip traces using more images than this (keeps CPU runs small)")
    ap.add_argument("--work-dir", default=None, help="Keep artefacts here (default: a temp dir)")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in open(args.traces, encoding="utf-8") if line.strip()]
    chosen = select_traces(records, args.num_traces, args.max_images)
    if not chosen:
        print("[SMOKE] no trace satisfies --max-images; raise it.")
        return 2

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="sft_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    model_dir, data_dir, out_dir = work / "model", work / "data", work / "out"
    model_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    print(f"[SMOKE] building tiny Qwen3.5 + tokenizer in {model_dir} ...")
    build_tiny_model(records, model_dir)
    trace_path = data_dir / "smoke_traces.jsonl"
    trace_path.write_text("\n".join(json.dumps(r) for r in write_fake_images(chosen, data_dir)) + "\n", encoding="utf-8")

    cmd = [
        sys.executable, "-W", "ignore", str(repo_root / "scripts" / "train_sft.py"),
        "--model-id", str(model_dir), "--track", "with_tools", "--stage", "dentex_alone",
        "--dataset-path", str(trace_path), "--data-dir", str(data_dir), "--output-dir", str(out_dir),
        "--epochs", "1", "--batch-size", "1", "--gradient-accumulation-steps", "2",
        "--lora-r", "4", "--lora-alpha", "8", "--learning-rate", "1e-3", "--precision", "bf16",
        *passthrough,
    ]
    print("[SMOKE] " + " ".join(cmd))
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    code = subprocess.call(cmd, env=env)
    ok = code == 0 and (out_dir / "adapter_config.json").exists()
    print(f"[SMOKE] {'PASSED' if ok else 'FAILED'} (exit code {code}; adapter saved: {(out_dir / 'adapter_config.json').exists()})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
