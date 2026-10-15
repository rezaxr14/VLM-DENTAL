"""Static checks of a GRPO command, run before the model is loaded (``scripts/run_grpo.py --preflight``).

A GRPO run costs hours and a TPU session; most of the ways it dies or silently trains on the wrong thing can be seen in
seconds: no SFT adapter where the script looks, no base model or YOLO checkpoint, a different canonical-resize setting than the SFT run, a missing dev
split, image files that are not on disk, a resume that no longer matches the pool, no hub token so checkpoints are never
uploaded, a TPU with a different core count than ``--num-cores``. Nothing here changes any setting; it only reports. It never
downloads; it only LISTS Hub repos (read-only) to tell "fetched at start-up" from "nowhere to be found".

``run_sft_preflight`` does the same for ``scripts/train_sft.py`` (traces, trace images, token-length manifest, dev split).

Levels: FAIL = the run would stop or train on the wrong thing; WARN = probably not what was intended; INFO = a fact.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

from dental_agent.training.run_meta import keys_fingerprint, read_sft_run_meta
from dental_agent.utils import inputs as inp
from dental_agent.utils.canonical import parse_slot_budget, slot_totals, tokens_per_image

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


def _flat(e: BaseException) -> str:
    return " | ".join(str(e).splitlines())


def _find_sft_dir(args: Any, candidates: list[Path]) -> Path | None:
    """An explicit --sft-model-dir is never replaced by another folder (the run would not do it either)."""
    if args.sft_model_dir:
        for c in inp.as_candidates(args.sft_model_dir):
            if c.is_dir() and any(c.iterdir()):
                return c
        return None
    for c in candidates:
        for cc in inp.as_candidates(c):
            if inp.has_adapter_config(cc):
                return cc
    for c in candidates:  # a non-empty folder without adapter files: reported precisely by _check_sft_adapter
        for cc in inp.as_candidates(c):
            if cc.is_dir() and any(cc.iterdir()):
                return cc
    return None


def _check_sft_adapter(args: Any, candidates: list[Path], rep: _Report) -> Path | None:
    found = _find_sft_dir(args, candidates)
    if found is None:
        if args.sft_model_dir:
            rep.add(FAIL, "SFT adapter", f"--sft-model-dir {args.sft_model_dir} does not exist; an explicit folder is never "
                    "replaced by another adapter. train_sft.py writes data/models/qwen3_5_9b_sft_<track>_<stage>_<precision>.")
            return None
        searched = ", ".join(str(c) for c in candidates)
        if args.hf_repo:
            names = list(dict.fromkeys([inp.sft_adapter_name(args.track, args.sft_stage, getattr(args, "precision", None)),
                                        inp.sft_adapter_name(args.track, args.sft_stage)]))
            present, err = inp.hub_probe(args.hf_repo, "model", [f"sft/{n}/adapter_config.json" for n in names])
            if present:
                rep.add(WARN, "SFT adapter", f"not found locally (searched {searched}); {present[0].rsplit('/', 1)[0]} exists on "
                        f"{args.hf_repo} and is downloaded at start-up.")
            elif err:
                rep.add(WARN, "SFT adapter", f"not found locally (searched {searched}) and the Hub could not be checked ({err}); "
                        "run_grpo tries the Hub at start-up and stops before loading the model if it is not there.")
            else:
                rep.add(FAIL, "SFT adapter", f"not found locally (searched {searched}) and {args.hf_repo} has no "
                        f"sft/{names[0]}/. Train it (scripts/train_sft.py) or pass --sft-model-dir.")
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
        if os.environ.get("DENTEX_IMAGES_REPO"):
            rep.add(WARN, "dataset", f"DENTEX train data not found under {cfg.data_dir}; the annotations and the images are fetched "
                    f"from {os.environ['DENTEX_IMAGES_REPO']} at start-up, so the pool cannot be checked here.")
        else:
            rep.add(FAIL, "dataset", f"DENTEX train data not found under {cfg.data_dir}. Download it first "
                    "(the preflight does not download) or set DENTEX_IMAGES_REPO to the per-image Hub repo.")
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

    # Rows with no file at all were dropped from the pool without a word; count them and see whether the Hub holds them.
    if "local_path" in images_df.columns:
        unmapped = images_df[images_df["local_path"].isna()]
        if len(unmapped):
            fixed, chk = inp.ensure_dataset_images(unmapped, dataset, cfg.data_dir, hub="off")
            if chk.n_missing:
                fetchable, unavailable, err = inp.probe_images_on_hub(chk.missing)
                if err:
                    rep.add(WARN, "image files", f"{chk.describe()}; the Hub could not be checked ({err}).")
                elif unavailable:
                    rep.add(FAIL, "image files", f"{chk.describe()}; {unavailable} are not on the Hub either "
                            "(python scripts/fetch_hf_assets.py --images).")
                else:
                    rep.add(WARN, "image files", f"{chk.describe()}; all {fetchable} are fetched from the Hub at start-up.")
            else:
                rep.add(INFO, "image files", f"{len(unmapped)} images had no path from the loader; the search finds them all.")
            images_df = images_df.copy()
            images_df.loc[unmapped.index, "local_path"] = fixed["local_path"]

    try:
        dev_keys = load_dev_keys(inp.resolve_dev_split(args.dev_split, cfg.data_dir))
    except (FileNotFoundError, inp.InputNotFound) as e:
        rep.add(FAIL, "dev split", _flat(e) if isinstance(e, inp.InputNotFound) else str(e))
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
        if args.spmd and getattr(args, "group_size", None) is not None and args.group_size < n:
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


def _check_base_model(args: Any, rep: _Report) -> None:
    model_id = getattr(args, "model_id", None)
    if not model_id:
        return
    try:
        r = inp.resolve_base_model(model_id, hub="probe")
    except inp.InputNotFound as e:
        rep.add(FAIL, "base model", _flat(e))
        return
    if r.source in ("local", "hub-cache"):
        rep.add(PASS, "base model", f"{r.path} ({'attached/local folder' if r.source == 'local' else 'Hugging Face cache'})")
    else:
        rep.add(WARN, "base model", f"{model_id} is not on this machine; it is downloaded from the Hub at start-up (about 18 GB). "
                "On Kaggle attach the model input and pass its folder as --model-id to skip that.")


def _check_yolo(args: Any, rep: _Report) -> None:
    """locate_tooth needs the YOLO checkpoint; a missing one only showed up as error replies during rollouts."""
    if getattr(args, "track", "with_tools") != "with_tools":
        return
    try:
        r = inp.resolve_yolo_checkpoint(hub="off")
        rep.add(PASS, "YOLO grounding checkpoint", f"{r.path} ({r.source})")
        return
    except inp.InputNotFound as local_err:
        searched = ", ".join(local_err.searched[:3])
    present, err = inp.hub_probe(inp.models_repo(getattr(args, "hf_repo", None)), "model", [inp.YOLO_HUB_FILE])
    if present:
        rep.add(WARN, "YOLO grounding checkpoint", f"not on this machine (searched {searched}); fetched from the Hub at start-up.")
    elif err:
        rep.add(WARN, "YOLO grounding checkpoint", f"not on this machine (searched {searched}) and the Hub could not be checked ({err}).")
    else:
        rep.add(FAIL, "YOLO grounding checkpoint", f"not found (searched {searched}) and the Hub has no {inp.YOLO_HUB_FILE}. "
                "Fix: python scripts/fetch_hf_assets.py --models, or set GROUNDING_MODEL_PATH.")


def _check_slot_waste(args: Any, track: str, rep: _Report) -> None:
    """Static slot padding fills every sample to the slot budget; a no_tools trace only ever shows one FULL image."""
    if track != "no_tools" or not getattr(args, "pad_vision_to_slots", False):
        return
    budget = list(args.vision_slots)
    have = slot_totals(parse_slot_budget(*budget))["tokens"]
    need = tokens_per_image("FULL")
    if have > need:
        rep.add(WARN, "vision slots (no_tools)", f"--vision-slots {' '.join(map(str, budget))} pads {have} vision tokens per sample, but a "
                f"no_tools trace shows one FULL image = {need} tokens ({have - need} of the {have} are padding). "
                "--vision-slots 1 0 0 matches this track. Nothing was changed.")


def _count_lines(p: Path) -> int:
    with open(p, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _check_sft_traces(args: Any, rep: _Report) -> list[Path]:
    if args.dataset_path:
        files = [Path(x.strip()) for x in args.dataset_path.split(",") if x.strip()]
        missing = [str(f) for f in files if not f.is_file()]
        if missing:
            rep.add(FAIL, "traces", f"--dataset-path file(s) not found: {missing}")
            return []
    else:
        try:
            res = inp.resolve_stage_traces(args.stage, args.track, args.data_dir, hub="probe")
        except inp.InputNotFound as e:
            rep.add(FAIL, "traces", _flat(e))
            return []
        files = res.files
        if res.hub_pending:
            rep.add(WARN, "traces", f"{res.hub_pending} are not local; they are downloaded from {inp.traces_repo()} at start-up "
                    "(only these files).")
        if res.legacy_fallback:
            rep.add(WARN, "traces", f"no stage files anywhere: the run would train on the legacy single file {files[0].name} "
                    f"(all cohorts), not stage '{args.stage}'.")
    if files:
        rep.add(PASS, "traces", ", ".join(f"{f.name} ({_count_lines(f)})" for f in files))
    return files


def _check_track_parity(files: list[Path], rep: _Report) -> None:
    """Every image should have a trace in BOTH tracks (the 18 Tufts no-tools traces were missing)."""
    if not files:
        return
    names = {f.name for f in files}
    for row in inp.trace_parity(files[0].parent):
        base = row["file"]
        twin = base.replace(".jsonl", "_no_tools.jsonl")
        if base not in names and twin not in names:
            continue
        gap = row["only_with_tools"]
        if gap:
            rep.add(WARN, "track parity", f"{base}: {len(gap)} image(s) have a with-tools trace but no no-tools trace "
                    f"(e.g. {', '.join(gap[:3])}); the no_tools track trains on {row['no_tools']} of {row['with_tools']}. "
                    "Regenerate: python scripts/run_trace_gen.py --mode generate --no-tools (then --mode verify --no-tools).")
        if row["only_no_tools"]:
            rep.add(INFO, "track parity", f"{base}: {len(row['only_no_tools'])} image(s) only have a no-tools trace.")


def _check_trace_images(files: list[Path], args: Any, dev_keys: set[str], rep: _Report) -> None:
    if not files:
        return
    chk = inp.check_trace_images(files, args.data_dir, exclude_keys=dev_keys, hub="off")
    if not chk.n_missing:
        rep.add(PASS, "trace images", f"all {chk.n_unique} resolve (working-directory independent)")
        return
    fetchable, unavailable, err = inp.probe_images_on_hub(chk.missing)
    detail = chk.describe()
    if err:
        rep.add(WARN, "trace images", f"{detail}; the Hub could not be checked ({err}).")
    elif unavailable:
        rep.add(FAIL, "trace images", f"{detail}; {unavailable} of them are not on the Hub either. "
                "Fix: python scripts/fetch_hf_assets.py --images (or attach the Kaggle image dataset).")
    else:
        rep.add(WARN, "trace images", f"{detail}; all {fetchable} are fetched from the Hub at start-up.")


def _check_manifest(files: list[Path], args: Any, rep: _Report) -> None:
    if not files:
        return
    st = inp.check_token_manifest(args.data_dir, files, args.canonical_resize)
    if st.state == "fresh":
        rep.add(PASS, "token-length manifest", f"{st.path} matches the current code and traces")
        return
    cmd = inp.manifest_command(args.data_dir, args.canonical_resize, args.model_id, args.vision_slots)
    level = FAIL if args.pad_vision_to_slots else WARN
    rep.add(level, "token-length manifest", f"{st.state}: {'; '.join(st.reasons)}. Recompute: {cmd}"
            + ("" if args.pad_vision_to_slots else " (without it over-length traces are not filtered)"))


def _check_sft_flags(args: Any, rep: _Report) -> None:
    ok = True
    if args.pad_vision_to_slots and not args.canonical_resize:
        rep.add(FAIL, "pad-vision-to-slots", "requires --canonical-resize.")
        ok = False
    if args.pad_vision_to_slots:
        tokens = slot_totals(parse_slot_budget(*args.vision_slots))["tokens"]
        if args.max_seq_len <= tokens:
            rep.add(FAIL, "max-seq-len", f"{args.max_seq_len} cannot hold the {tokens} static vision tokens of --vision-slots "
                    f"{' '.join(map(str, args.vision_slots))}.")
            ok = False
    if args.xla_spmd and importlib.util.find_spec("torch_xla") is not None and not args.fsdp:
        rep.add(FAIL, "fsdp", "--spmd shards the weights through FSDPv2; --no-fsdp would replicate 18 GB on every core.")
        ok = False
    if ok:
        rep.add(PASS, "flags", "combination is consistent")
    if args.max_seq_len != 16384:
        rep.add(INFO, "max-seq-len", f"{args.max_seq_len} (the project default is 16384)")


def _check_sft_output(args: Any, rep: _Report) -> Path:
    out = Path(args.output_dir or f"data/models/qwen3_5_9b_sft_{args.track}_{args.stage}_{args.precision}")
    if out.is_dir() and inp.has_adapter_config(out):
        if args.resume_hf:
            rep.add(INFO, "output folder", f"{out} already holds an adapter; --resume-hf overlays the Hub checkpoint on it.")
        else:
            rep.add(WARN, "output folder", f"{out} already holds an adapter; this run (no --resume-hf) starts at step 1 and overwrites it.")
    else:
        rep.add(PASS, "output folder", f"{out}")
    if args.resume_hf:
        present, err = inp.hub_probe(args.resume_hf, "model", [f"sft/{out.name}/training_state.json"])
        if present:
            rep.add(PASS, "resume", f"{args.resume_hf}/sft/{out.name} holds a checkpoint to continue from")
        elif err:
            rep.add(WARN, "resume", f"the Hub could not be checked ({err}).")
        else:
            rep.add(WARN, "resume", f"{args.resume_hf} has no sft/{out.name}/training_state.json: the run starts at step 1.")
    return out


def run_sft_preflight(args: Any, emit: Callable[[str], None] = print) -> int:
    """Static checks of a ``train_sft.py`` command, before any model is loaded. Returns the number of FAILs (0 = ready)."""
    from dental_agent.data.dev_split import load_dev_keys

    rep = _Report(emit)
    emit(f"SFT preflight: track={args.track} stage={args.stage} precision={args.precision} data-dir={args.data_dir}")
    _check_sft_flags(args, rep)
    _check_base_model(args, rep)
    files = _check_sft_traces(args, rep)

    dev_keys: set[str] = set()
    try:
        dev_path = inp.resolve_dev_split(args.dev_split, args.data_dir)
        dev_keys = load_dev_keys(dev_path)
        rep.add(PASS if dev_path else WARN, "dev split",
                f"{len(dev_keys)} held-out images ({dev_path})" if dev_path else "none given: no images are held out for the K sweep / ablations; "
                "pass --dev-split data/splits/dentex_dev.json (the same file as GRPO).")
    except inp.InputNotFound as e:
        rep.add(FAIL, "dev split", _flat(e))

    _check_trace_images(files, args, dev_keys, rep)
    _check_track_parity(files, rep)
    _check_manifest(files, args, rep)
    _check_slot_waste(args, args.track, rep)
    out = _check_sft_output(args, rep)

    hub_view = Namespace(hf_repo=args.hf_repo, push_every_steps=args.push_every_steps, precision=args.precision,
                         num_cores=args.num_cores, spmd=args.xla_spmd, pad_vision_to_slots=args.pad_vision_to_slots)
    _check_hub(hub_view, rep)
    _check_hardware(hub_view, rep)
    _check_disk(out, rep)
    emit(f"Preflight result: {rep.n_fail} FAIL, {rep.n_warn} WARN" + ("  -> fix the FAILs before launching." if rep.n_fail else "  -> ready."))
    return rep.n_fail


def run_preflight(args: Any, sft_candidates: list[Path], emit: Callable[[str], None] = print) -> int:
    """Run every check for the parsed ``run_grpo.py`` arguments. Returns the number of FAILs (0 = ready to launch)."""
    from dental_agent.config import load_config

    rep = _Report(emit)
    emit(f"GRPO preflight: track={args.track} K={args.group_size} stage={args.sft_stage} dataset={args.dataset}")
    cfg = load_config(args.config)

    _check_flags(args, rep)
    _check_base_model(args, rep)
    _check_yolo(args, rep)
    _check_slot_waste(args, args.track, rep)
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
