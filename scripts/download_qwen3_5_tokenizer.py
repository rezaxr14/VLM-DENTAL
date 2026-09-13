#!/usr/bin/env python3
"""
scripts/download_qwen3_5_tokenizer.py

Surgical downloader for authentic Qwen 3.5 tokenizer, chat template, and processor configs.
Fetches only lightweight tokenizer/preprocessor assets (~16 MB total) from Hugging Face Hub
(default: Qwen/Qwen3.5-9B) into data/qwen3_5_tokenizer/ without downloading model weights.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

# Ensure repo root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(repo_root / ".env")

TARGET_REPO = os.environ.get("QWEN3_5_MODEL_ID", "Qwen/Qwen3.5-9B")
TARGET_DIR = repo_root / "data" / "qwen3_5_tokenizer"

ESSENTIAL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
]


def download_file(repo_id: str, filename: str, out_dir: Path) -> Path:
    """Download a single file from Hugging Face Hub with fallback to raw resolve URL."""
    out_path = out_dir / filename
    if out_path.is_file() and out_path.stat().st_size > 0:
        print(f"  [CACHE] {filename} already exists ({out_path.stat().st_size:,} bytes). Skipping.")
        return out_path

    # Method 1: Try huggingface_hub if available
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        print(f"  [FETCH:hf_hub] Downloading {filename} from {repo_id}...")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )
        p = Path(downloaded)
        print(f"    --> Saved: {filename} ({p.stat().st_size:,} bytes)")
        return p
    except Exception as e:
        print(f"  [WARN:hf_hub] hf_hub_download failed for {filename} ({e}). Trying raw HTTPS stream...")

    # Method 2: Direct raw resolve URL stream
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    headers = {"User-Agent": "VLM-Dental-Tokenizer-Downloader/1.0"}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    req = urllib.request.Request(url, headers=headers)
    print(f"  [FETCH:https] GET {url}...")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as out_f:
            total_bytes = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_f.write(chunk)
                total_bytes += len(chunk)
        print(f"    --> Saved: {filename} ({total_bytes:,} bytes)")
        return out_path
    except Exception as err:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"Failed to download {filename} from {repo_id}: {err}") from err


def main():
    print("=" * 70)
    print(f"QWEN 3.5 TOKENIZER SURGICAL INGESTION")
    print(f"Source Model : {TARGET_REPO}")
    print(f"Target Dir   : {TARGET_DIR}")
    print("=" * 70)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for fn in ESSENTIAL_FILES:
        try:
            download_file(TARGET_REPO, fn, TARGET_DIR)
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] {fn}: {e}")

    print("\n" + "-" * 70)
    print(f"Downloaded {success_count}/{len(ESSENTIAL_FILES)} files to {TARGET_DIR}")
    print("-" * 70)

    # Verification: Try loading AutoProcessor and AutoTokenizer from local files
    print("\n[VERIFICATION] Instantiating tokenizer and processor from local snapshot...")
    try:
        from transformers import AutoTokenizer, AutoProcessor
        tokenizer = AutoTokenizer.from_pretrained(str(TARGET_DIR), local_files_only=True)
        print(f"  [OK] AutoTokenizer loaded successfully! Vocab size: {len(tokenizer):,}")
        print(f"       Pad token: {tokenizer.pad_token!r} (ID: {tokenizer.pad_token_id})")
        print(f"       EOS token: {tokenizer.eos_token!r} (ID: {tokenizer.eos_token_id})")

        processor = AutoProcessor.from_pretrained(str(TARGET_DIR), local_files_only=True, trust_remote_code=True)
        print(f"  [OK] AutoProcessor loaded successfully! Class: {processor.__class__.__name__}")
        if hasattr(processor, "image_processor"):
            img_p = processor.image_processor
            print(f"       Image Processor: {img_p.__class__.__name__} (patch_size={getattr(img_p, 'patch_size', 'N/A')}, merge_size={getattr(img_p, 'merge_size', 'N/A')})")

        # Verify chat template
        test_messages = [
            {"role": "system", "content": "You are a dental radiologist AI."},
            {"role": "user", "content": [{"type": "text", "text": "Analyze image."}]},
        ]
        rendered = processor.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
        print(f"  [OK] Chat template formatted successfully! Preview:\n{rendered[:160]}...\n")
        print(">> ALL VERIFICATIONS PASSED: Authentic Qwen 3.5 Tokenizer is ready for local/cloud pipelines!")
    except Exception as e:
        print(f"  [WARN] Verification failed or partial: {e}")
        print("  (If transformers version does not yet have native Qwen3VLProcessor registered locally,")
        print("   tokenizer and raw config files are still cleanly saved for Colab/TPU environments).")


if __name__ == "__main__":
    main()
