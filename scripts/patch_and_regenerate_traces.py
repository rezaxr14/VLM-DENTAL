#!/usr/bin/env python3
"""
scripts/patch_and_regenerate_traces.py

End-to-end, foolproof trace regeneration, dual-gate verification,
canonical splicing, and Hugging Face Hub synchronization.

Key Invariants:
1. ID-Based Surgical Purge: Drops only known contaminated image IDs, leaving all
   other verified traces 100% untouched (no loose regex across clean traces).
2. Dual-Gate Quality Filter on New Generations:
   - Gate 1: Zero-Leak Audit (rejects any assistant turn mentioning directives/hints/GT).
   - Gate 2: Clinical Ground-Truth Verifier via MiniMax M3 (openrouter / minimax/minimax-m3:free).
3. Resilient While-Loop: Loops over missing target IDs until 100% of images are verified.
4. Canonical Splicing: Reconstructs train_cot_traces.jsonl and train_cot_traces_no_tools.jsonl.
5. Hugging Face Hub Persist: upload_traces(force=True) to replace remote files.
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

from dental_agent.config import load_config, load_env
from dental_agent.data.dentex import load_dentex_dataset, download_dentex_slice
from dental_agent.data.tufts import load_tufts_dataset, load_tufts_normal_dataset, download_tufts_slice
from dental_agent.training.trace_generation import generate_only, generate_only_no_tools, verify_trace
from dental_agent.utils.serialization import to_jsonable
from scripts.sync_traces_hf import upload_traces

load_env()

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


def get_existing_ids(file_path: Path) -> set[int]:
    """Load valid completed image IDs from a JSONL file."""
    if not file_path.exists():
        return set()
    completed = set()
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    completed.add(int(json.loads(line)["image_id"]))
                except Exception:
                    pass
    return completed


def purge_infected_ids_from_file(file_path: Path, infected_ids: set[int]) -> int:
    """Surgically drop lines matching exact infected image IDs. Preserves all other lines."""
    if not file_path.exists():
        return 0

    clean_lines = []
    removed_count = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                img_id = int(rec.get("image_id", -1))
                if img_id in infected_ids:
                    removed_count += 1
                    continue
            except Exception:
                pass
            clean_lines.append(line_str)

    if removed_count > 0:
        tmp_path = file_path.with_suffix(".tmp_purge")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for l in clean_lines:
                f.write(l + "\n")
        tmp_path.replace(file_path)

    return removed_count


def splice_canonical_files(trace_dir: Path) -> None:
    """Rebuild canonical merged files from updated, clean split files."""
    print("\n" + "=" * 70)
    print("SPLICING CANONICAL MERGED TRACE FILES")
    print("=" * 70)

    # 1. train_cot_traces.jsonl = DENTEX + Tufts
    dentex_p = trace_dir / "train_cot_traces_dentex.jsonl"
    tufts_p = trace_dir / "train_cot_traces_tufts.jsonl"
    can_tools_p = trace_dir / "train_cot_traces.jsonl"

    dentex_lines = [l.strip() for l in open(dentex_p, "r", encoding="utf-8") if l.strip()] if dentex_p.exists() else []
    tufts_lines = [l.strip() for l in open(tufts_p, "r", encoding="utf-8") if l.strip()] if tufts_p.exists() else []

    with open(can_tools_p, "w", encoding="utf-8") as f:
        for l in dentex_lines + tufts_lines:
            f.write(l + "\n")
    print(f"* Canonical With-Tools: {len(dentex_lines)} DENTEX + {len(tufts_lines)} Tufts = {len(dentex_lines) + len(tufts_lines)} total -> {can_tools_p.name}")

    # 2. train_cot_traces_no_tools.jsonl = DENTEX No-Tools + Tufts No-Tools
    dentex_nt_p = trace_dir / "train_cot_traces_dentex_no_tools.jsonl"
    tufts_nt_p = trace_dir / "train_cot_traces_tufts_no_tools.jsonl"
    can_nt_p = trace_dir / "train_cot_traces_no_tools.jsonl"

    dentex_nt_lines = [l.strip() for l in open(dentex_nt_p, "r", encoding="utf-8") if l.strip()] if dentex_nt_p.exists() else []
    tufts_nt_lines = [l.strip() for l in open(tufts_nt_p, "r", encoding="utf-8") if l.strip()] if tufts_nt_p.exists() else []

    with open(can_nt_p, "w", encoding="utf-8") as f:
        for l in dentex_nt_lines + tufts_nt_lines:
            f.write(l + "\n")
    print(f"* Canonical No-Tools  : {len(dentex_nt_lines)} DENTEX + {len(tufts_nt_lines)} Tufts = {len(dentex_nt_lines) + len(tufts_nt_lines)} total -> {can_nt_p.name}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Looping Trace Regeneration & Verification (MiniMax M3)")
    parser.add_argument("--trace-dir", type=str, default="data/traces", help="Directory containing trace files")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum while-loop rounds to resolve missing traces")
    parser.add_argument("--pacing-delay", type=float, default=2.0, help="Delay (seconds) between successive image API calls")
    parser.add_argument("--provider", type=str, default="openrouter", help="Generator and verifier provider")
    parser.add_argument("--model", type=str, default="minimax/minimax-m3:free", help="Generator and verifier model")
    parser.add_argument("--skip-upload", action="store_true", help="Skip final upload to Hugging Face Hub")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    data_dir = getattr(cfg, "data_dir", os.environ.get("DENTAL_AGENT_DATA_DIR", "data"))

    provider = args.provider
    model = os.environ.get("OPENROUTER_GENERATOR_MODEL") or args.model
    hf_repo = os.environ.get("HF_TRACES_REPO", "Reza-Nadimi/vlm-dental-traces")

    print("\n" + "=" * 70)
    print("VLM-DENTAL: ROBUST LOOPING TRACE GENERATOR & VERIFIER")
    print("=" * 70)
    print(f"* Provider         : {provider}")
    print(f"* Model            : {model}")
    print(f"* Trace Directory  : {trace_dir}")
    print(f"* Max Loop Rounds  : {args.max_rounds}")
    print(f"* Pacing Delay     : {args.pacing_delay}s")
    print(f"* HF Repository    : {hf_repo}")
    print("=" * 70 + "\n")

    # Step 1: Pre-Filter Known Infected IDs (Protects against contaminated start-of-session HF download)
    print("[1/4] Ensuring all local split files are stripped of contaminated IDs...")
    for c in TARGET_CONFIGS:
        fpath = trace_dir / c["split_file"]
        dropped = purge_infected_ids_from_file(fpath, set(c["target_ids"]))
        if dropped > 0:
            print(f"  [Purged] {fpath.name}: dropped {dropped} contaminated record(s).")
        else:
            print(f"  [OK]     {fpath.name}: clean of target infected IDs.")

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
        slice_repo = os.environ.get(repo_env)

        round_num = 1
        while round_num <= args.max_rounds:
            existing = get_existing_ids(target_file)
            unresolved = [i for i in target_ids if i not in existing]

            if not unresolved:
                print(f"  --> [RESOLVED] All {len(target_ids)} images for {batch_name} successfully verified!")
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
                    imgs_df["local_path"] = imgs_df.apply(lambda r: str(local_map.get(r["id"], r["local_path"])), axis=1)

            # Process pending images
            for idx, image_id in enumerate(unresolved, start=1):
                print(f"\n  ({idx}/{len(unresolved)}) Processing Image ID {image_id}...")
                dataset_key = "tufts_all" if c["all_diseases"] else dataset_name
                success = False

                for attempt in range(1, 4):
                    if attempt > 1 or args.pacing_delay > 0:
                        time.sleep(args.pacing_delay)

                    # Generation Call
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
                            max_turns=25,
                            max_tokens_per_turn=16384,
                            healthy_only=c["healthy_only"],
                            provider=provider,
                            model=model,
                            dataset=dataset_key,
                        )

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

                        with open(target_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(to_jsonable(traj)) + "\n")

                        print(f"    [Gate 1 & Gate 2 PASSED] Image {image_id} verified & appended to {target_file.name}")
                        success = True
                        break
                    else:
                        v_reason = (v_res.get("reason") or "unspecified rejection")[:80]
                        print(f"    [Gate 2 REJECT] Attempt {attempt}: {v_reason}")

                if not success:
                    print(f"    [NOTICE] Image ID {image_id} unresolved in Round {round_num}.")

            round_num += 1

    # Step 3: Canonical Splicing
    print("\n[3/4] Rebuilding canonical merged trace files...")
    splice_canonical_files(trace_dir)

    # Step 4: Hugging Face Upload
    if not args.skip_upload:
        print("\n[4/4] Uploading clean, spliced traces to Hugging Face Hub...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token or hf_token.startswith("your_"):
            print("  ⚠️ HF_TOKEN not configured. Skipping remote upload.")
        else:
            upload_traces(repo_id=hf_repo, source_dir=str(trace_dir), force=True)
            print(f"  [OK] Successfully pushed patched datasets to {hf_repo}")
    else:
        print("\n[4/4] Skipping upload (--skip-upload set).")

    print("\n" + "=" * 70)
    print("END-TO-END PATCH & REGENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
