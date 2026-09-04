#!/usr/bin/env python3
"""
scripts/sync_traces_hf.py

Bidirectional synchronization between local synthetic traces (data/traces/*.jsonl)
and Hugging Face Hub dataset repository (default: Reza-Nadimi/vlm-dental-traces).

Features:
- --upload: Uploads local traces to Hugging Face dataset repo with retry handling.
- --download: Surgical download of missing traces only (Rule 16: Local Asset Re-Use).
- --status: Compares local files vs remote Hugging Face repository.
- --target-dir: Custom directory for downloads (enables safe, non-destructive testing).
"""

import argparse
import os
import sys
import time
from pathlib import Path
import dotenv

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

dotenv.load_dotenv()

DEFAULT_TRACES_REPO = os.environ.get("HF_TRACES_REPO", "Reza-Nadimi/vlm-dental-traces")

# Canonical completed trace files
CANONICAL_TRACE_FILES = [
    "train_cot_traces.jsonl",
    "train_cot_traces_no_tools.jsonl",
    "train_cot_traces_dentex.jsonl",
    "train_cot_traces_dentex_no_tools.jsonl",
    "train_cot_traces_tufts.jsonl",
    "train_cot_traces_tufts_no_tools.jsonl",
    "train_cot_traces_healthy_dentex.jsonl",
    "train_cot_traces_healthy_dentex_no_tools.jsonl",
    "train_cot_traces_healthy_tufts.jsonl",
    "train_cot_traces_healthy_tufts_no_tools.jsonl",
    "train_cot_traces_unverified_dentex.jsonl",
    "train_cot_traces_unverified_dentex_no_tools.jsonl",
    "train_cot_traces_unverified_healthy_dentex.jsonl",
    "train_cot_traces_unverified_healthy_dentex_no_tools.jsonl",
    "train_cot_traces_unverified_healthy_tufts.jsonl",
    "train_cot_traces_unverified_healthy_tufts_no_tools.jsonl",
    "train_cot_traces_unverified_tufts.jsonl",
    "train_cot_traces_unverified_tufts_no_tools.jsonl",
    "train_cot_traces_tufts_all.jsonl",
    "train_cot_traces_tufts_all_no_tools.jsonl",
    "train_cot_traces_unverified_tufts_all.jsonl",
    "train_cot_traces_unverified_tufts_all_no_tools.jsonl",
]


def get_hf_api(token: str | None = None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)
    
    t = token or os.environ.get("HF_TOKEN")
    if not t or t.startswith("your_"):
        t = None
    return HfApi(token=t)


def upload_traces(
    repo_id: str = DEFAULT_TRACES_REPO,
    source_dir: str | Path = "data/traces",
    public: bool = False,
    token: str | None = None,
    files: list[str] | None = None,
    force: bool = False,
) -> None:
    """Uploads completed trace files from source_dir to Hugging Face dataset repo."""
    api = get_hf_api(token)
    source_path = Path(source_dir)

    print(f"\n========================================================")
    print(f"HUGGING FACE TRACE UPLOAD")
    print(f"Target Repo: {repo_id} (dataset, private={not public})")
    print(f"Source Dir : {source_path}")
    print(f"========================================================")

    # 1. Ensure repository exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=not public, exist_ok=True)
        print(f"[OK] Repository {repo_id} verified.")
    except Exception as e:
        print(f"[WARNING] Could not create/verify repo {repo_id} directly ({e}). Proceeding with upload attempts...")

    # Fetch remote files for surgical delta upload (Rule 16)
    try:
        from huggingface_hub import list_repo_files
        t = token or os.environ.get("HF_TOKEN")
        if not t or t.startswith("your_"):
            t = None
        remote_files = set(list_repo_files(repo_id=repo_id, repo_type="dataset", token=t))
    except Exception as e:
        print(f"[WARNING] Could not list remote repo files ({e}).")
        remote_files = set()

    # 2. Collect files to upload
    target_names = files if files else CANONICAL_TRACE_FILES
    local_files = [source_path / f for f in target_names if (source_path / f).exists()]
    if not local_files and not files:
        # Fallback to any jsonl in source_path
        local_files = list(source_path.glob("*.jsonl"))

    print(f"Found {len(local_files)} trace files candidate for upload:")
    total_bytes = sum(f.stat().st_size for f in local_files)
    print(f"Total candidate payload: {total_bytes / (1024 * 1024):.2f} MB\n")

    # 3. Upload files with retry logic
    max_retries = 5
    uploaded = 0
    skipped = 0
    for idx, f in enumerate(local_files, 1):
        rel_name = f.name
        fsize_mb = f.stat().st_size / (1024 * 1024)

        if not force and rel_name in remote_files:
            print(f"[{idx}/{len(local_files)}] [CACHED] {rel_name} ({fsize_mb:.2f} MB) already on HF. Skipping.", flush=True)
            skipped += 1
            continue

        print(f"[{idx}/{len(local_files)}] Uploading {rel_name} ({fsize_mb:.2f} MB)...", flush=True)

        for attempt in range(max_retries):
            try:
                api.upload_file(
                    path_or_fileobj=str(f),
                    path_in_repo=rel_name,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"Upload synthetic trace: {rel_name} ({fsize_mb:.2f} MB)",
                )
                print(f"  --> Success!")
                uploaded += 1
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
                if attempt < max_retries - 1:
                    wait_s = 60 * (attempt + 1) if is_rate_limit else 5 * (attempt + 1)
                    print(f"  [Notice] Upload error ({e}). Retrying in {wait_s}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_s)
                else:
                    print(f"  [FAILED] Could not upload {rel_name}: {e}")

    print(f"\nUpload pass complete: {uploaded} uploaded, {skipped} skipped (already present) out of {len(local_files)} files to {repo_id}.\n")


def download_traces(
    repo_id: str = DEFAULT_TRACES_REPO,
    target_dir: str | Path = "data/traces",
    token: str | None = None,
    files: list[str] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Surgically downloads trace files from Hugging Face dataset repo if missing locally."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    t = token or os.environ.get("HF_TOKEN")
    if not t or t.startswith("your_"):
        t = None

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================================")
    print(f"HUGGING FACE TRACE DOWNLOAD")
    print(f"Source Repo: {repo_id} (dataset)")
    print(f"Target Dir : {target_path}")
    print(f"========================================================")

    # 1. Discover available remote files
    try:
        remote_files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=t)
        remote_jsonl = [f for f in remote_files if f.endswith(".jsonl")]
    except Exception as e:
        print(f"[ERROR] Could not inspect files in {repo_id}: {e}")
        return {}

    target_files = files if files else remote_jsonl
    results = {}

    for fname in target_files:
        if fname not in remote_jsonl:
            continue
        dest_file = target_path / fname

        # Rule 16: Surgical local caching check
        if dest_file.exists() and dest_file.stat().st_size > 0 and not force:
            print(f"[EXISTS] {fname} ({dest_file.stat().st_size / (1024*1024):.2f} MB) already present. Skipping.")
            results[fname] = "cached"
            continue

        print(f"[FETCH] Downloading {fname} from {repo_id}...", flush=True)
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type="dataset",
                token=t,
                local_dir=str(target_path),
            )
            print(f"  --> Saved to {dest_file}")
            results[fname] = "downloaded"
        except Exception as e:
            print(f"  --> Failed to download {fname}: {e}")
            results[fname] = f"error: {e}"

    print(f"Trace download pass complete: {sum(1 for v in results.values() if v in ('cached', 'downloaded'))}/{len(results)} files ready.\n")
    return results


def status_traces(
    repo_id: str = DEFAULT_TRACES_REPO,
    source_dir: str | Path = "data/traces",
    token: str | None = None,
) -> None:
    """Displays synchronization status between local directory and Hugging Face."""
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        print("ERROR: huggingface_hub is required.")
        return

    t = token or os.environ.get("HF_TOKEN")
    if not t or t.startswith("your_"):
        t = None

    source_path = Path(source_dir)
    print(f"\nChecking status for {repo_id} vs {source_path}...")

    try:
        remote_files = set(list_repo_files(repo_id=repo_id, repo_type="dataset", token=t))
    except Exception as e:
        print(f"Could not connect to Hugging Face repo {repo_id}: {e}")
        remote_files = set()

    local_files = {f.name: f for f in source_path.glob("*.jsonl")}

    all_names = sorted(set(CANONICAL_TRACE_FILES) | set(local_files.keys()) | {f for f in remote_files if f.endswith(".jsonl")})

    print(f"{'Trace File':<50} | {'Local Disk':<15} | {'Hugging Face':<15}")
    print("-" * 86)
    for name in all_names:
        l_stat = f"{local_files[name].stat().st_size / (1024*1024):.2f} MB" if name in local_files else "MISSING"
        r_stat = "PRESENT" if name in remote_files else "MISSING"
        print(f"{name:<50} | {l_stat:<15} | {r_stat:<15}")
    print("-" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Synchronize synthetic traces with Hugging Face Hub dataset repo.")
    parser.add_argument("--upload", action="store_true", help="Upload local traces to Hugging Face dataset repo.")
    parser.add_argument("--download", action="store_true", help="Download missing traces from Hugging Face dataset repo.")
    parser.add_argument("--status", action="store_true", help="Compare local trace files against Hugging Face repository.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_TRACES_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_TRACES_REPO})")
    parser.add_argument("--target-dir", type=str, default="data/traces", help="Target directory for downloads (default: data/traces)")
    parser.add_argument("--source-dir", type=str, default="data/traces", help="Source directory for uploads (default: data/traces)")
    parser.add_argument("--public", action="store_true", help="Set repository to public (default: private)")
    parser.add_argument("--files", nargs="+", help="Specific files to upload or download.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files on download or upload.")
    args = parser.parse_args()

    if args.upload:
        upload_traces(repo_id=args.repo_id, source_dir=args.source_dir, public=args.public, files=args.files, force=args.force)
    elif args.download:
        download_traces(repo_id=args.repo_id, target_dir=args.target_dir, files=args.files, force=args.force)
    elif args.status:
        status_traces(repo_id=args.repo_id, source_dir=args.source_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
