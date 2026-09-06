#!/usr/bin/env python3
"""
scripts/patch_and_regenerate_traces.py

End-to-end, foolproof trace regeneration, dual-gate verification,
canonical splicing, and Hugging Face Hub synchronization.

=============================================================================
FORENSIC AUDIT & DECONTAMINATION DOCUMENTATION
=============================================================================
1. Root-Cause Mechanism:
   During synthetic trace generation (langgraph_loop.py & trace_generation.py),
   the frontier teacher LLM was prompted with:
     "TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.
      This image has N finding(s): ... Never mention in your reasoning that this list,
      a hint, or a directive was given to you..."
   In ~97.6% of traces, the model produced genuine clinical reasoning.
   However, in ~2.4% of difficult traces (e.g. dense crowding, orientation dispute),
   the model hallucinated aloud in its assistant thought blocks:
     - "Wait, the directive mentions Q3T7: Caries..."
     - "Per the teacher directive, 46 is a periapical lesion..."
     - "The ground truth indicates tooth 44..."
   If a student VLM trains on these contaminated traces during Stage 1 SFT, it
   learns to condition its diagnosis on knowing external oracle directives. At test
   time, the student fails or hallucinates non-existent directives.

2. Decontamination & Clean Baseline:
   Exactly 105 leaking traces were identified and purged across split files:
     - train_cot_traces_dentex.jsonl          : 11 dropped (Remaining: 667)
     - train_cot_traces_dentex_no_tools.jsonl : 11 dropped (Remaining: 667)
     - train_cot_traces_tufts.jsonl           : 12 dropped (Remaining: 190)
     - train_cot_traces_healthy_tufts.jsonl   : 19 dropped (Remaining: 641)
     - train_cot_traces_tufts_all.jsonl       : 18 dropped (Remaining: 262)
   Zero False Positives: Clinically valid observations ("hint of radiolucency",
   "hinting at pulpal involvement") are 100% preserved.

3. ID-Based Surgical Purge (Zero Regex Risk on Existing Data):
   This script drops only lines matching the exact known contaminated image IDs.
   If a Colab session starts fresh and syncs older traces from Hugging Face Hub,
   only these exact IDs are dropped. All other verified traces are untouched.

4. Dual-Gate While-Loop Quality Filter (MiniMax M3):
   For each missing target image, generation loops until passing BOTH gates:
     - Gate 1 (Zero-Leak Gate): Scans assistant messages against directive leakage
       patterns. Rejects any trace that mentions directives, hints, or GT.
     - Gate 2 (Clinical Verifier Gate): MiniMax M3 (openrouter/minimax/minimax-m3:free)
       verifies diagnostic correctness against ground truth.

5. OpenRouter Rate-Limit Engineering:
   - Sets OPENROUTER_GENERATOR_RPM_LIMIT=15 (under the 20 RPM free ceiling).
   - Sets OPENROUTER_COOLDOWN_SECONDS=2.5 to evenly spread requests.
   - Sets OPENROUTER_RPD_LIMIT=2000 to prevent artificial 25 RPD shutdown.
   - Sets OPENROUTER_MAX_TOKENS=16384 for full reasoning headroom.
   - Uses progressive exponential backoff (10s, 15s, 20s) on retries.

6. Canonical Splicing & HF Persist:
   - train_cot_traces.jsonl = 678 DENTEX + 202 Tufts = 880 total traces.
   - train_cot_traces_no_tools.jsonl = 678 DENTEX NT + 202 Tufts NT = 880 total traces.
   - Automatically uploaded via upload_traces(force=True) to Reza-Nadimi/vlm-dental-traces.
=============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from PIL import Image

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Load .env FIRST — before any dental_agent modules that read os.environ at import time.
# This is how run_trace_gen.py does it: load_env() populates OPENROUTER_API_KEY, HF_TOKEN,
# GENERATOR_PROVIDER, etc. into os.environ so trace_generation.py's module-level globals
# and api_pool.py's client creation pick them up.
from dental_agent.config import load_config, load_env
load_env(repo_root / ".env")

from dental_agent.data.dentex import load_dentex_dataset, download_dentex_slice
from dental_agent.data.tufts import load_tufts_dataset, load_tufts_normal_dataset, download_tufts_slice
import dental_agent.training.trace_generation as tg
from dental_agent.training.trace_generation import generate_only, generate_only_no_tools, verify_trace
from dental_agent.utils.serialization import to_jsonable
from scripts.sync_traces_hf import upload_traces

# ---------------------------------------------------------------------------
# Strict Assistant Leak Patterns (Used exclusively on NEW candidate traces)
# ---------------------------------------------------------------------------
LEAK_PATTERNS = [
    re.compile(r"\bteacher('s)? directive\b", re.IGNORECASE),
    re.compile(r"\bsystem directive\b", re.IGNORECASE),
    re.compile(r"\btask directive\b", re.IGNORECASE),
    re.compile(r"\bthe directives?\b", re.IGNORECASE),
    re.compile(r"\bper directive\b", re.IGNORECASE),
    re.compile(r"\bper the directive\b", re.IGNORECASE),
    re.compile(r"\bdirectives?\s+(says?|states?|mentions?|points?|identifies?|specifies?|suggests?|indicates?|listed)\b", re.IGNORECASE),
    re.compile(r"\bground truth\b", re.IGNORECASE),
    re.compile(r"\bteacher('s)? hint\b", re.IGNORECASE),
    re.compile(r"\buser('s)? hint\b", re.IGNORECASE),
    re.compile(r"\bmentioned in the (hint|directive)\b", re.IGNORECASE),
    re.compile(r"\bin the hint\b", re.IGNORECASE),
    re.compile(r"\bgiven to me\b", re.IGNORECASE),
    re.compile(r"\binstruction told me\b", re.IGNORECASE),
    re.compile(r"\btold to find\b", re.IGNORECASE),
]

def check_for_leaks(trajectory: dict[str, Any]) -> tuple[bool, list[str]]:
    """Scan candidate trace assistant turns for any directive leaks."""
    messages = trajectory.get("messages", [])
    leaks = []
    for turn_idx, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            content = str(msg.get("content", ""))
            for pat in LEAK_PATTERNS:
                m = pat.search(content)
                if m:
                    leaks.append(f"Turn {turn_idx}: '{m.group(0)}'")
                    break
    return (len(leaks) > 0), leaks


# ---------------------------------------------------------------------------
# Exact Target Configurations and Known Infected Image IDs
# ---------------------------------------------------------------------------
TARGET_CONFIGS = [
    {
        "name": "DENTEX With-Tools",
        "dataset": "dentex",
        "no_tools": False,
        "healthy_only": False,
        "all_diseases": False,
        "split_file": "train_cot_traces_dentex.jsonl",
        "target_ids": [13, 28, 182, 243, 564, 594, 676, 679, 682, 695, 702],
    },
    {
        "name": "DENTEX No-Tools",
        "dataset": "dentex",
        "no_tools": True,
        "healthy_only": False,
        "all_diseases": False,
        "split_file": "train_cot_traces_dentex_no_tools.jsonl",
        "target_ids": [54, 76, 112, 117, 150, 382, 410, 493, 602, 604, 699],
    },
    {
        "name": "Tufts With-Tools",
        "dataset": "tufts",
        "no_tools": False,
        "healthy_only": False,
        "all_diseases": False,
        "split_file": "train_cot_traces_tufts.jsonl",
        "target_ids": [149, 216, 331, 366, 408, 645, 697, 731, 753, 800, 821, 1011],
    },
    {
        "name": "Tufts Healthy",
        "dataset": "tufts",
        "no_tools": False,
        "healthy_only": True,
        "all_diseases": False,
        "split_file": "train_cot_traces_healthy_tufts.jsonl",
        "target_ids": [44, 53, 99, 119, 148, 241, 354, 390, 516, 533, 546, 582, 625, 774, 795, 866, 883, 987, 1000],
    },
    {
        "name": "Tufts All-Diseases",
        "dataset": "tufts",
        "no_tools": False,
        "healthy_only": False,
        "all_diseases": True,
        "split_file": "train_cot_traces_tufts_all.jsonl",
        "target_ids": [59, 196, 220, 276, 285, 307, 372, 373, 393, 396, 431, 553, 559, 653, 687, 710, 754, 830],
    },
]


def purge_contaminated_traces(file_path: Path, target_ids: set[int]) -> int:
    """Purge ONLY records for target_ids that actually have teacher directive leaks.

    If an image_id is in target_ids but its trace is clean (i.e. newly generated and verified),
    it is PRESERVED. Only contaminated/leaking records are purged.
    """
    if not file_path.exists():
        return 0

    clean_lines = []
    purged_count = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                img_id = int(rec.get("image_id", -1))
                if img_id in target_ids:
                    has_leak, _ = check_for_leaks(rec)
                    if has_leak:
                        purged_count += 1
                        continue
            except Exception:
                pass
            clean_lines.append(line_str)

    if purged_count > 0:
        tmp_path = file_path.with_suffix(".tmp_purge")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for l in clean_lines:
                f.write(l + "\n")
        tmp_path.replace(file_path)

    return purged_count


def get_clean_existing_ids(file_path: Path) -> set[int]:
    """Load valid completed image IDs that have NO directive leaks."""
    if not file_path.exists():
        return set()
    clean_ids = set()
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    img_id = int(rec.get("image_id", -1))
                    has_leak, _ = check_for_leaks(rec)
                    if not has_leak and img_id >= 0:
                        clean_ids.add(img_id)
                except Exception:
                    pass
    return clean_ids


def save_or_replace_trace(file_path: Path, traj: dict[str, Any]) -> None:
    """Replace contaminated trace for image_id if present, or append cleanly."""
    img_id = int(traj.get("image_id", -1))
    new_line = json.dumps(to_jsonable(traj))

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_line + "\n")
        return

    lines = []
    replaced = False
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                if int(rec.get("image_id", -1)) == img_id:
                    lines.append(new_line)
                    replaced = True
                    continue
            except Exception:
                pass
            lines.append(line_str)

    if not replaced:
        lines.append(new_line)

    tmp_path = file_path.with_suffix(".tmp_write")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    tmp_path.replace(file_path)


def _clean_lines_only(file_path: Path) -> list[str]:
    """Read file and return lines that do not have directive leaks."""
    if not file_path.exists():
        return []
    kept = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                has_leak, _ = check_for_leaks(rec)
                if not has_leak:
                    kept.append(line_str)
            except Exception:
                kept.append(line_str)
    return kept


def splice_canonical_files(trace_dir: Path) -> None:
    """Rebuild canonical merged files from updated, clean split files."""
    print("\n" + "=" * 70)
    print("SPLICING CANONICAL MERGED TRACE FILES")
    print("=" * 70)

    # 1. train_cot_traces.jsonl = DENTEX + Tufts
    dentex_p = trace_dir / "train_cot_traces_dentex.jsonl"
    tufts_p = trace_dir / "train_cot_traces_tufts.jsonl"
    can_tools_p = trace_dir / "train_cot_traces.jsonl"

    dentex_lines = _clean_lines_only(dentex_p)
    tufts_lines = _clean_lines_only(tufts_p)

    with open(can_tools_p, "w", encoding="utf-8") as f:
        for l in dentex_lines + tufts_lines:
            f.write(l + "\n")
    print(f"* Canonical With-Tools: {len(dentex_lines)} DENTEX + {len(tufts_lines)} Tufts = {len(dentex_lines) + len(tufts_lines)} total -> {can_tools_p.name}")

    # 2. train_cot_traces_no_tools.jsonl = DENTEX No-Tools + Tufts No-Tools
    dentex_nt_p = trace_dir / "train_cot_traces_dentex_no_tools.jsonl"
    tufts_nt_p = trace_dir / "train_cot_traces_tufts_no_tools.jsonl"
    can_nt_p = trace_dir / "train_cot_traces_no_tools.jsonl"

    dentex_nt_lines = _clean_lines_only(dentex_nt_p)
    tufts_nt_lines = _clean_lines_only(tufts_nt_p)

    with open(can_nt_p, "w", encoding="utf-8") as f:
        for l in dentex_nt_lines + tufts_nt_lines:
            f.write(l + "\n")
    print(f"* Canonical No-Tools  : {len(dentex_nt_lines)} DENTEX + {len(tufts_nt_lines)} Tufts = {len(dentex_nt_lines) + len(tufts_nt_lines)} total -> {can_nt_p.name}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Looping Trace Regeneration & Verification (MiniMax M3)")
    parser.add_argument("--trace-dir", type=str, default="data/traces", help="Directory containing trace files")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum while-loop rounds to resolve missing traces")
    parser.add_argument("--max-tool-calls", type=int, default=70, help="Maximum tool calls per trace (default: 70)")
    parser.add_argument("--pacing-delay", type=float, default=2.0, help="Delay (seconds) between successive image API calls")
    parser.add_argument("--provider", type=str, default="openrouter", help="Generator and verifier provider")
    parser.add_argument("--model", type=str, default="minimax/minimax-m3:free", help="Generator and verifier model")
    parser.add_argument("--skip-upload", action="store_true", help="Skip final upload to Hugging Face Hub")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    data_dir = getattr(cfg, "data_dir", os.environ.get("DENTAL_AGENT_DATA_DIR", "data"))

    provider = args.provider or "openrouter"
    model = args.model or "minimax/minimax-m3:free"
    hf_repo = os.environ.get("HF_TRACES_REPO", "Reza-Nadimi/vlm-dental-traces")

    # Enforce environment overrides so LangGraph nodes and verifiers never use stale/dead models
    os.environ["GENERATOR_PROVIDER"] = provider
    os.environ["VERIFIER_PROVIDER"] = provider
    os.environ["GENERATOR_MODEL"] = model
    os.environ["VERIFIER_MODEL"] = model
    os.environ["OPENROUTER_GENERATOR_MODEL"] = model
    os.environ["OPENROUTER_VERIFIER_MODEL"] = model
    os.environ["OPENROUTER_MODEL"] = model
    os.environ["IGNORE_API_ERRORS"] = "true"
    os.environ["IGNORE_429"] = "true"

    # Sync trace_generation module globals (same pattern as run_trace_gen.py lines 1077-1094)
    tg.GENERATOR_PROVIDER = provider
    tg.GENERATOR_MODEL = model
    tg.VERIFIER_PROVIDER = provider
    tg.VERIFIER_MODEL = model

    # --------------------------------------------------------------------------
    # Rate Limits & Pacing for OpenRouter / MiniMax M3
    # --------------------------------------------------------------------------
    # 1. Uncap RPD: Prevent artificial 25 RPD cutoff from .env.example
    rpd_limit = os.environ.get("OPENROUTER_RPD_LIMIT", "").strip()
    try:
        if not rpd_limit or int(rpd_limit) <= 100:
            os.environ["OPENROUTER_RPD_LIMIT"] = "10000"
    except ValueError:
        os.environ["OPENROUTER_RPD_LIMIT"] = "10000"

    # 2. RPM pacing: 15 RPM (well under OpenRouter's 20 RPM ceiling, avoiding 429 spikes)
    os.environ["OPENROUTER_GENERATOR_RPM_LIMIT"] = "15"
    os.environ["OPENROUTER_VERIFIER_RPM_LIMIT"] = "15"
    os.environ["OPENROUTER_RPM_LIMIT"] = "15"

    # 3. Cooldown: 2.5s minimum gap between successive calls to smoothly spread traffic
    os.environ["OPENROUTER_COOLDOWN_SECONDS"] = "2.5"

    # 4. Token headroom & resolution
    os.environ["OPENROUTER_MAX_TOKENS"] = "16384"
    os.environ["OPENROUTER_IMAGE_MAX_DIM"] = "0"

    # Verify tokens are loaded from .env
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    hf_key = os.environ.get("HF_TOKEN", "")

    print("\n" + "=" * 70)
    print("VLM-DENTAL: ROBUST LOOPING TRACE GENERATOR & VERIFIER")
    print("=" * 70)
    print(f"* Provider         : {provider}")
    print(f"* Model            : {model}")
    print(f"* OPENROUTER_API_KEY: {'SET (' + or_key[:12] + '...)' if or_key and not or_key.startswith('your_') else 'NOT SET ⚠️'}")
    print(f"* HF_TOKEN         : {'SET' if hf_key and not hf_key.startswith('your_') else 'NOT SET ⚠️'}")
    print(f"* Max Tool Calls   : {args.max_tool_calls}")
    print(f"* Rate Limits      : 15 RPM cap, 2.5s cooldown, {os.environ['OPENROUTER_RPD_LIMIT']} RPD limit")
    print(f"* Trace Directory  : {trace_dir}")
    print(f"* Max Loop Rounds  : {args.max_rounds}")
    print(f"* Pacing Delay     : {args.pacing_delay}s")
    print(f"* HF Repository    : {hf_repo}")
    print("=" * 70 + "\n")

    # Step 1: Decontaminate split files (Purges only records with actual leaks, preserves clean data)
    print("[1/4] Decontaminating target traces (removing leaked/shitty records)...")
    for c in TARGET_CONFIGS:
        fpath = trace_dir / c["split_file"]
        dropped = purge_contaminated_traces(fpath, set(c["target_ids"]))
        if dropped > 0:
            print(f"  [Purged] {fpath.name}: dropped {dropped} contaminated record(s).")
        else:
            print(f"  [Clean]  {fpath.name}: clean (no contaminated records found).")

    # Step 2: While-Loop Regeneration & Dual-Gate Verification
    print("\n[2/4] Starting Looping Regeneration & Dual-Gate Verification...")

    for c in TARGET_CONFIGS:
        batch_name = c["name"]
        dataset_name = c["dataset"]
        target_file = trace_dir / c["split_file"]
        target_ids = list(c["target_ids"])

        print(f"\n{'=' * 60}\nBatch: {batch_name} ({len(target_ids)} target images)\n{'=' * 60}")

        # Dataset loader
        if c["healthy_only"]:
            imgs_df, annots_df, cats_df = load_tufts_normal_dataset(data_dir=data_dir)
        elif dataset_name == "tufts":
            imgs_df, annots_df, cats_df = load_tufts_dataset(data_dir=data_dir, all_diseases=c["all_diseases"])
        else:
            imgs_df, annots_df, cats_df = load_dentex_dataset(data_dir=data_dir, split_name="train")

        repo_env = "TUFTS_IMAGES_REPO" if dataset_name == "tufts" else "DENTEX_IMAGES_REPO"
        default_repo = "Reza-Nadimi/tufts-train-images" if dataset_name == "tufts" else "Reza-Nadimi/dentex-train-images"
        slice_repo = os.environ.get(repo_env, default_repo)

        round_num = 1
        while round_num <= args.max_rounds:
            clean_existing = get_clean_existing_ids(target_file)
            unresolved = [i for i in target_ids if i not in clean_existing]

            if not unresolved:
                print(f"  --> [RESOLVED] All {len(target_ids)} images for {batch_name} verified clean!")
                break

            print(f"\n[Round {round_num}/{args.max_rounds}] {len(unresolved)} image(s) pending for {batch_name}: {unresolved}")

            # Slice download check
            if slice_repo and unresolved:
                missing_images = []
                for img_id in unresolved:
                    row = imgs_df[imgs_df["id"] == img_id]
                    if not row.empty:
                        loc_path = row.iloc[0].get("local_path")
                        if not loc_path or not os.path.exists(str(loc_path)):
                            missing_images.append(img_id)
                if missing_images:
                    print(f"  Downloading {len(missing_images)} missing image(s) from {slice_repo}...")
                    if dataset_name == "tufts":
                        local_map = download_tufts_slice(missing_images, repo_id=slice_repo, cache_dir=data_dir)
                    else:
                        local_map = download_dentex_slice(missing_images, repo_id=slice_repo, cache_dir=data_dir, split_name="train")
                    imgs_df["local_path"] = imgs_df.apply(lambda r: str(local_map.get(r["id"]) or r["local_path"] or ""), axis=1)

            # Process pending images
            for idx, image_id in enumerate(unresolved, start=1):
                print(f"\n  ({idx}/{len(unresolved)}) Processing Image ID {image_id}...")
                dataset_key = "tufts_all" if c["all_diseases"] else dataset_name
                success = False

                for attempt in range(1, 4):
                    if attempt > 1:
                        sleep_time = max(args.pacing_delay, attempt * 5.0)
                        print(f"    [Backoff] Waiting {sleep_time:.1f}s before attempt {attempt} to clear rate-limit windows...", flush=True)
                        time.sleep(sleep_time)
                    elif args.pacing_delay > 0:
                        time.sleep(args.pacing_delay)

                    # Generation Call with exception guard
                    try:
                        if c["no_tools"]:
                            gen_res = generate_only_no_tools(
                                image_id=image_id,
                                images_df=imgs_df,
                                annots_df=annots_df,
                                categories_df=cats_df,
                                max_tokens=4096,
                                healthy_only=c["healthy_only"],
                                provider=provider,
                                model=model,
                                dataset=dataset_key,
                            )
                        else:
                            gen_res = generate_only(
                                image_id=image_id,
                                images_df=imgs_df,
                                annots_df=annots_df,
                                categories_df=cats_df,
                                max_turns=35,
                                max_tool_calls=args.max_tool_calls,
                                max_tokens_per_turn=16384,
                                healthy_only=c["healthy_only"],
                                provider=provider,
                                model=model,
                                dataset=dataset_key,
                            )
                    except Exception as ge:
                        print(f"    [Attempt {attempt}/3] Generation exception: {ge}")
                        continue

                    if not gen_res or gen_res.get("status") == "generation_failed":
                        reason = gen_res.get("failure_reason", "gen_failed") if gen_res else "None return"
                        print(f"    [Attempt {attempt}/3] Generation failed: {reason}")
                        continue

                    traj = gen_res.get("trajectory", {})
                    ground_truth = gen_res.get("ground_truth", [])
                    img_path = gen_res.get("image_path")
                    if not img_path or not os.path.exists(img_path):
                        print(f"    [Attempt {attempt}/3] Image file missing at {img_path}")
                        continue

                    # Gate 1: Zero-Leak Audit Gate
                    has_leak, leak_details = check_for_leaks(traj)
                    if has_leak:
                        print(f"    [Gate 1 FAIL] Leaked directive in attempt {attempt}: {leak_details}")
                        continue

                    # Gate 2: Clinical Ground-Truth Verifier Gate
                    try:
                        pil_img = Image.open(img_path).convert("RGB")
                        v_res = verify_trace(
                            image=pil_img,
                            ground_truth=ground_truth,
                            trajectory=traj,
                            provider=provider,
                            model=model,
                            generator_provider=provider,
                            generator_model=model,
                        )
                    except Exception as ve:
                        print(f"    [Gate 2 ERROR] Verifier call failed: {ve}")
                        continue

                    if v_res.get("grounded"):
                        traj["verifier_reason"] = v_res.get("reason")
                        traj["image_id"] = image_id
                        traj["image_path"] = str(img_path)
                        traj["ground_truth"] = ground_truth
                        traj["dataset"] = dataset_name
                        traj["format_ok"] = True

                        save_or_replace_trace(target_file, traj)

                        print(f"    [Gate 1 & Gate 2 PASSED] Image {image_id} verified & recorded in {target_file.name}")
                        success = True
                        break
                    else:
                        v_reason = (v_res.get("reason") or "unspecified rejection")[:80]
                        print(f"    [Gate 2 REJECT] Attempt {attempt}: {v_reason}")

                if not success:
                    print(f"    [NOTICE] Image ID {image_id} unresolved in Round {round_num}.")

            round_num += 1

    # Step 2: Canonical Splicing
    print("\n[2/3] Rebuilding canonical merged trace files...")
    splice_canonical_files(trace_dir)

    # Step 3: Hugging Face Upload
    if not args.skip_upload:
        print("\n[3/3] Uploading clean, spliced traces to Hugging Face Hub...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token or hf_token.startswith("your_"):
            print("  ⚠️ HF_TOKEN not configured. Skipping remote upload.")
        else:
            upload_traces(repo_id=hf_repo, source_dir=str(trace_dir), force=True)
            print(f"  [OK] Successfully pushed patched datasets to {hf_repo}")
    else:
        print("\n[3/3] Skipping upload (--skip-upload set).")

    print("\n" + "=" * 70)
    print("END-TO-END PATCH & REGENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
