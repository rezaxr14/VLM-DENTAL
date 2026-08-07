"""
Test Aim 1 Synthetic CoT Trace Generation on real DENTEX sample images.
"""

from __future__ import annotations

import json
import os
import sys
import time
from PIL import Image

from dental_agent.config import load_config, load_env
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.training.api_pool import get_gemini_pool, call_llm
from dental_agent.training.trace_generation import (
    generate_trace,
    verify_trace,
    GENERATOR_PROVIDER,
    GENERATOR_MODEL,
    VERIFIER_PROVIDER,
    VERIFIER_MODEL,
)


def main() -> None:
    load_env()
    cfg = load_config()

    print("=" * 70, flush=True)
    print("AIM 1: SYNTHETIC CoT TRACE GENERATION & VERIFICATION DEMO", flush=True)
    print("=" * 70, flush=True)

    pool = get_gemini_pool()
    print(f"Loaded {len(pool.keys)} Gemini API keys in pool.", flush=True)
    print(f"Configured models: {pool.models}", flush=True)
    print(f"Generator: {GENERATOR_PROVIDER}/{GENERATOR_MODEL}", flush=True)
    print(f"Verifier:  {VERIFIER_PROVIDER}/{VERIFIER_MODEL}", flush=True)

    print("\n1. Loading DENTEX validation split...", flush=True)
    imgs_df, annots_df, cats_df = load_dentex_dataset(data_dir=cfg.data_dir, split_name="validation")
    print(f"Loaded {len(imgs_df)} validation images and {len(annots_df)} annotations.", flush=True)

    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    if valid_imgs.empty:
        print("Error: No images found with valid local paths.", flush=True)
        sys.exit(1)

    target_img_id = int(valid_imgs.iloc[0]["id"])
    target_path = valid_imgs.iloc[0]["local_path"]
    anns = annots_df[annots_df["image_id"] == target_img_id]
    ann0 = anns.iloc[0]
    cat_lookup = dict(zip(cats_df["id"], cats_df["name"])) if not cats_df.empty else {}

    ground_truth = {
        "quadrant": int(ann0.get("category_id_1", 1)),
        "tooth_position": int(ann0.get("category_id_2", 1)),
        "diagnosis": cat_lookup.get(ann0.get("category_id_3"), "Caries"),
        "bbox": list(ann0.get("bbox", [0, 0, 50, 50])),
    }

    print(f"\n2. Ground Truth for Image {os.path.basename(target_path)} (ID {target_img_id}):", flush=True)
    print(json.dumps(ground_truth, indent=2), flush=True)

    image = Image.open(target_path).convert("RGB")

    print("\n3. Calling Generator (Gemini) to create step-by-step reasoning trace...", flush=True)
    t0 = time.time()
    candidates = generate_trace(image, ground_truth, k=1)
    gen_time = time.time() - t0
    print(f"Candidate generated in {gen_time:.2f}s:\n", flush=True)
    print("-------------------- GENERATED CoT TRACE --------------------", flush=True)
    print(candidates[0], flush=True)
    print("-------------------------------------------------------------", flush=True)

    print("\n4. Calling Verifier to judge groundedness...", flush=True)
    t1 = time.time()
    v_res = verify_trace(image, ground_truth, candidates[0])
    ver_time = time.time() - t1
    print(f"Verifier completed in {ver_time:.2f}s: {v_res}", flush=True)

    print("\n--- UPDATED KEY POOL STATUS ---", flush=True)
    print(pool.status().head(6).to_string(index=False), flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
