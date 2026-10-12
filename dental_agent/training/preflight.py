"""Static checks of a GRPO command, run before the model is loaded (``scripts/run_grpo.py --preflight``).

A GRPO run costs hours and a TPU session; most of the ways it dies or silently trains on the wrong thing can be seen in
seconds: no SFT adapter where the script looks, a different canonical-resize setting than the SFT run, a missing dev
split, image files that are not on disk, a resume that no longer matches the pool, no hub token so checkpoints are never
uploaded, a TPU with a different core count than ``--num-cores``. Nothing here changes any setting; it only reports.

Levels: FAIL = the run would stop or train on the wrong thing; WARN = probably not what was intended; INFO = a fact.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from dental_agent.training.run_meta import keys_fingerprint, read_sft_run_meta
from dental_agent.utils.canonical import parse_slot_budget, slot_totals

PASS, INFO, WARN, FAIL = "PASS", "INFO", "WARN", "FAIL"


class _Report:
    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.rows.append((level, name, detail))
        self.emit(f"[{level}] {name}" + (f": {detail}" if detail else ""))

    @property
    def n_fail(self) -> int:
        return sum(1 for level, _, _ in self.rows if level == FAIL)

    @property
    def n_warn(self) -> int:
        return sum(1 for level, _, _ in self.rows if level == WARN)


def _find_sft_dir(args: Any, candidates: list[Path]) -> Path | None:
    order = ([Path(args.sft_model_dir)] if args.sft_model_dir else []) + list(candidates)
    for c in order:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


def _check_sft_adapter(args: Any, candidates: list[Path], rep: _Report) -> Path | None:
    found = _find_sft_dir(args, candidates)
    if found is None:
        searched = ", ".join(str(c) for c in ([Path(args.sft_model_dir)] if args.sft_model_dir else []) + candidates)
        if args.hf_repo:
            rep.add(WARN, "SFT adapter", f"not found locally (searched {searched}); run_grpo will try to download it from "
                    f"{args.hf_repo}. Pass --sft-model-dir to use a local folder (train_sft.py writes "
                    "data/models/qwen3_5_9b_sft_<track>_<stage>_<precision> by default).")
        else:
            rep.add(FAIL, "SFT adapter", f"not found (searched {searched}). Pass --sft-model-dir "
                    "(train_sft.py writes data/models/qwen3_5_9b_sft_<track>_<stage>_<precision> by default).")
        return None
    has_cfg = (found / "adapter_config.json").is_file()
    has_w = (found / "adapter_model.safetensors").is_file() or (found / "adapter_model.bin").is_file()
    if not (has_cfg and has_w):
        hint = " (it has a best_adapter/ subfolder: pass that folder or copy it up)" if (found / "best_adapter").is_dir() else ""
        rep.add(FAIL, "SFT adapter", f"{found} has no adapter_config.json + adapter weights at its top level{hint}.")
        return found
    rep.add(PASS, "SFT adapter", str(found))
    return found


def _check_sft_consistency(args: Any, sft_dir: Path, dev_keys: set[str], rep: _Report) -> None:
    meta = read_sft_run_meta(sft_dir)
    if meta is None:
        rep.add(WARN, "SFT run metadata", f"{sft_dir} has no sft_run_meta.json (adapter predates it); canonical-resize, "
                "dev split and track cannot be compared with the SFT run. Check them against the SFT notebook by hand.")
        return
    if meta.get("track") not in (None, args.track):
        rep.add(FAIL, "track vs SFT", f"SFT adapter was trained for '{meta['track']}', GRPO is '{args.track}'.")
    if "canonical_resize" in meta and bool(meta["canonical_resize"]) != bool(args.canonical_resize):
        rep.add(FAIL, "canonical-resize vs SFT", f"SFT used canonical_resize={meta['canonical_resize']}, GRPO has "
                f"--canonical-resize={args.canonical_resize}: the policy would be optimised on a different image distribution.")
    else:
        rep.add(PASS, "canonical-resize vs SFT", f"both {bool(args.canonical_resize)}")
    sft_fp, grpo_fp = meta.get("dev_split_fingerprint"), keys_fingerprint(dev_keys)
    if sft_fp is not None and sft_fp != grpo_fp:
        rep.add(FAIL, "dev split vs SFT", f"SFT held out '{meta.get('dev_split')}' ({sft_fp}), GRPO holds out "
                f"'{args.dev_split}' ({grpo_fp}). Use the same --dev-split so no dev image is trained on by either stage.")
    elif sft_fp is not None:
        rep.add(PASS, "dev split vs SFT", "same held-out images" if dev_keys else "neither stage holds images out")
    if meta.get("max_seq_len") not in (None, args.max_seq_len):
        rep.add(WARN, "max-seq-len vs SFT", f"SFT {meta['max_seq_len']}, GRPO {args.max_seq_len}; use the SFT value.")
    if args.pad_vision_to_slots and meta.get("vision_slots") not in (None, list(args.vision_slots)):
        rep.add(WARN, "vision-slots vs SFT", f"SFT {meta['vision_slots']}, GRPO {list(args.vision_slots)}.")


def _check_flags(args: Any, rep: _Report) -> None:
    ok = True
    if args.temperature <= 0:
        rep.add(FAIL, "temperature", "must be > 0 (identical samples give zero advantages).")
        ok = False
    if args.resume_hf and not args.resume:
        rep.add(FAIL, "resume-hf", "only downloads the checkpoint; add --resume to continue from it.")
        ok = False
    if args.pad_vision_to_slots and not args.canonical_resize:
        rep.add(FAIL, "pad-vision-to-slots", "requires --canonical-resize.")
        ok = False
    if args.pad_vision_to_slots:
        tokens = slot_totals(parse_slot_budget(*args.vision_slots))["tokens"]
        if args.max_seq_len <= tokens:
            rep.add(FAIL, "max-seq-len", f"{args.max_seq_len} cannot hold the {tokens} static vision tokens of --vision-slots.")
            ok = False
    if ok:
        rep.add(PASS, "flags", "combination is consistent")
    if args.max_seq_len != 16384:
        rep.add(INFO, "max-seq-len", f"{args.max_seq_len} (the project default is 16384; use the SFT run's value)")


def _load_pool(args: Any, cfg: Any, rep: _Report):
    from dental_agent.data.dentex import find_local_dentex_dir, load_dentex_dataset
    from dental_agent.data.dev_split import image_key, load_dev_keys, select_training_pool
    from dental_agent.data.tufts import load_tufts_dataset

    dataset = args.dataset.strip().lower()
    if dataset != "tufts" and find_local_dentex_dir(cfg.data_dir, "train") is None:
        # The training loader would start a multi-GB download; a preflight must never do that.
        rep.add(FAIL, "dataset", f"DENTEX train data not found under {cfg.data_dir}. Download it first "
                "(the preflight does not download).")
        return None, set(), None
    try:
        if dataset == "tufts":
            images_df, annots_df, _ = load_tufts_dataset(cfg.data_dir)
        else:
            images_df, annots_df, _ = load_dentex_dataset(cfg.data_dir, split_name="train")
    except SystemExit as e:
        rep.add(FAIL, "dataset", str(e))
        return None, set(), None
    except Exception as e:
        rep.add(FAIL, "dataset", f"could not load {dataset} from {cfg.data_dir}: {e}")
        return None, set(), None

    try:
        dev_keys = load_dev_keys(args.dev_split)
    except FileNotFoundError as e:
        rep.add(FAIL, "dev split", str(e))
        return None, set(), None
    if not args.dev_split:
        rep.add(WARN, "dev split", "none given: nothing is held out for the K sweep and ablations. Pass the same "
                "--dev-split as SFT (scripts/make_dev_split.py).")
    else:
        train_keys = {image_key(dataset, i) for i in images_df["id"]}
        missing = sorted(dev_keys - train_keys)
        if missing:
            rep.add(FAIL, "dev split", f"{len(missing)} dev keys are not train images of '{dataset}' (e.g. {missing[:3]}).")
        else:
            rep.add(PASS, "dev split", f"{len(dev_keys)} images, all in the train split")

    try:
        pool = select_training_pool(
            images_df, annots_df, dataset=dataset, dev_keys=dev_keys,
            include_healthy=args.include_healthy, max_images=args.max_images, seed=args.seed,
        )
    except SystemExit as e:
        rep.add(FAIL, "train pool", str(e))
        return None, dev_keys, None
    overlap = [i for i in pool["id"] if image_key(dataset, i) in dev_keys]
    if overlap:
        rep.add(FAIL, "train pool", f"{len(overlap)} dev images are in the training pool.")
    annotated = set(annots_df["image_id"].unique()) if len(annots_df) else set()
    n_healthy = sum(1 for i in pool["id"] if i not in annotated)
    rep.add(PASS if len(pool) else FAIL, "train pool",
            f"{len(pool)} images = {len(pool)} GRPO steps ({n_healthy} healthy); seed {args.seed}")

    if "local_path" in pool.columns:
        missing_files = [p for p in pool["local_path"].dropna() if not Path(str(p)).is_file()]
        if missing_files:
            rep.add(FAIL, "image files", f"{len(missing_files)} of {len(pool)} pool images are not on disk (e.g. {missing_files[0]}).")
        else:
            rep.add(PASS, "image files", "every pool image exists")
    return pool, dev_keys, annots_df


def _check_resume(args: Any, pool: Any, out_dir: Path, rep: _Report) -> None:
    from dental_agent.training.grpo_resume import find_latest_step_checkpoint, pool_fingerprint

    latest = find_latest_step_checkpoint(out_dir, args.track)
    final = (out_dir / f"grpo-{args.track}-final").is_dir()
    if final:
        rep.add(INFO, "output folder", f"{out_dir} already holds grpo-{args.track}-final; --resume would finish immediately.")
    if not args.resume:
        if latest is not None:
            rep.add(WARN, "output folder", f"{out_dir} already holds {latest[0].name}; without --resume this run starts at "
                    "step 1 and overwrites those checkpoints.")
        else:
            rep.add(PASS, "output folder", f"{out_dir} (fresh run)")
        return
    if latest is None:
        if args.resume_hf:
            rep.add(WARN, "resume", f"no local step checkpoint yet; the newest one will be fetched from {args.resume_hf} at start-up "
                    "(the preflight does not contact the hub).")
        else:
            rep.add(WARN, "resume", f"--resume given but {out_dir} holds no step checkpoint: the run starts at step 1.")
        return
    path, meta = latest
    if pool is not None:
        fp = pool_fingerprint(pool.dropna(subset=["local_path"])["id"]) if "local_path" in pool.columns else pool_fingerprint(pool["id"])
        if meta.get("pool_fingerprint") != fp or int(meta.get("group_size", args.group_size)) != args.group_size:
            rep.add(FAIL, "resume", f"{path.name} was trained on another pool or group size (pool {meta.get('pool_fingerprint')} vs "
                    f"{fp}, K {meta.get('group_size')} vs {args.group_size}). Use the same --dev-split, --seed, --max-images, "
                    "--include-healthy, --group-size.")
            return
    rep.add(PASS, "resume", f"{path.name} matches this pool and K; continues at step {int(meta.get('step', 0)) + 1}")


def _check_hub(args: Any, rep: _Report) -> None:
    if not args.hf_repo:
        rep.add(WARN, "hub upload", "no --hf-repo: checkpoints stay on this machine only.")
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    if token:
        rep.add(PASS, "hub token", f"present; checkpoints upload to {args.hf_repo} every {args.push_every_steps} steps")
    else:
        rep.add(WARN, "hub token", f"none found: uploads to {args.hf_repo} will fail and only local checkpoints are kept. "
                "Set HF_TOKEN before starting if the session may be reclaimed.")


def _check_hardware(args: Any, rep: _Report) -> None:
    if importlib.util.find_spec("torch_xla") is not None:
        if args.precision == "qlora":
            rep.add(FAIL, "precision", "qlora is CUDA-only (bitsandbytes); use --precision bf16 on TPU.")
        try:
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            if args.spmd:
                import torch_xla.runtime as xr
                if hasattr(xr, "use_spmd"):
                    xr.use_spmd()
            import torch_xla.runtime as xr
            n = int(xr.global_runtime_device_count())
        except Exception as e:
            rep.add(WARN, "TPU", f"torch_xla is installed but the devices could not be counted: {e}")
            return
        if args.num_cores != n:
            rep.add(FAIL, "TPU cores", f"{n} devices visible but --num-cores is {args.num_cores}.")
        else:
            rep.add(PASS, "TPU cores", f"{n} devices, --num-cores {args.num_cores}")
        if args.spmd and args.group_size < n:
            rep.add(INFO, "group size", f"K={args.group_size} fills {args.group_size} of {n} rows in every policy forward "
                    f"(the rest are padding); K={n} uses every core for the same number of forwards.")
        if args.spmd and not args.pad_vision_to_slots:
            rep.add(WARN, "static shapes", "SPMD without --pad-vision-to-slots recompiles for every trajectory length.")
        return

    try:
        import torch
    except Exception as e:
        rep.add(FAIL, "torch", f"not importable: {e}")
        return
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rep.add(PASS, "GPU", f"{props.name}, {props.total_memory / 2**30:.0f} GB, --precision {args.precision}")
        if args.precision == "qlora" and importlib.util.find_spec("bitsandbytes") is None:
            rep.add(FAIL, "bitsandbytes", "--precision qlora needs bitsandbytes.")
        if args.pad_vision_to_slots:
            rep.add(FAIL, "pad-vision-to-slots", "TPU-only; GPU/CPU pad dynamically.")
    else:
        rep.add(WARN, "accelerator", "no CUDA GPU and no TPU runtime: GRPO on CPU is only useful for smoke tests.")


def _check_disk(out_dir: Path, rep: _Report) -> None:
    probe = out_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 2**30
    level = PASS if free_gb >= 5 else WARN
    rep.add(level, "disk", f"{free_gb:.0f} GB free at {probe}" + ("" if level == PASS else " (each step checkpoint is about 0.8 GB)"))


def run_preflight(args: Any, sft_candidates: list[Path], emit: Callable[[str], None] = print) -> int:
    """Run every check for the parsed ``run_grpo.py`` arguments. Returns the number of FAILs (0 = ready to launch)."""
    from dental_agent.config import load_config

    rep = _Report(emit)
    emit(f"GRPO preflight: track={args.track} K={args.group_size} stage={args.sft_stage} dataset={args.dataset}")
    cfg = load_config(args.config)

    _check_flags(args, rep)
    pool, dev_keys, _ = _load_pool(args, cfg, rep)
    sft_dir = _check_sft_adapter(args, sft_candidates, rep)
    if sft_dir is not None and (sft_dir / "adapter_config.json").is_file():
        _check_sft_consistency(args, sft_dir, dev_keys, rep)

    out_dir = Path(args.output_dir) / f"qwen3_5_9b_grpo_{args.track}_k{args.group_size}_{args.sft_stage}"
    _check_resume(args, pool, out_dir, rep)
    _check_hub(args, rep)
    _check_hardware(args, rep)
    _check_disk(out_dir, rep)

    emit(f"Preflight result: {rep.n_fail} FAIL, {rep.n_warn} WARN" + ("  -> fix the FAILs before launching." if rep.n_fail else "  -> ready."))
    return rep.n_fail
