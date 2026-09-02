import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO


def get_model_root() -> Path:
    """Return the directory used for YOLO artifacts. Allows notebook/local overrides."""
    override = os.environ.get("YOLO_MODELS_ROOT")
    if override:
        return Path(override).expanduser()
    return Path("data/models")


def _dataset_dir_suffix(datasets_arg: str) -> str:
    """Same naming convention as prepare_yolo_dataset.py's dir_suffix, so
    --datasets dentex,tufts here always points at the exact directory that
    script produced for the same value -- kept in one place so the two
    scripts can't silently drift apart."""
    if "," in datasets_arg:
        dataset_list = [d.strip() for d in datasets_arg.split(",") if d.strip()]
    else:
        dataset_list = [d.strip() for d in datasets_arg.split() if d.strip()]
    return "_".join(dataset_list) if dataset_list != ["dentex"] else "dentex"


def _model_subdir_prefix(dir_suffix: str) -> str:
    """Empty for default 'dentex', '{dir_suffix}_' for multi-dataset combos."""
    return "" if dir_suffix == "dentex" else f"{dir_suffix}_"


def train_single(yaml_path: str, args) -> dict:
    """Train a single YOLO model and return metrics."""
    model_root = get_model_root()
    prefix = _model_subdir_prefix(_dataset_dir_suffix(getattr(args, "datasets", "dentex")))
    resume = getattr(args, "resume", False)
    if resume:
        last_pt = model_root / f"{prefix}grounding_tool" / "weights" / "last.pt"
        if last_pt.exists():
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print("No checkpoint found — starting from scratch.")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    import torch
    device = getattr(args, "device", "0")
    if device in ("0", "cuda") and not torch.cuda.is_available():
        device = "cpu"

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(model_root),
        name=f"{prefix}grounding_tool",
        exist_ok=True,
    )
    if hasattr(args, "patience") and args.patience:
        train_kwargs["patience"] = args.patience

    print(f"Starting training on {device} for {args.epochs} epochs (patience={getattr(args, 'patience', 'N/A')})...")
    result = model.train(**train_kwargs)

    print("\nTraining Complete!")
    print(f"Model saved to: {model_root / f'{prefix}grounding_tool' / 'weights' / 'best.pt'}")

    return {}


def cross_validate(args):
    """Train YOLO independently on each CV fold, report metrics, select best."""
    model_root = get_model_root()
    dir_suffix = _dataset_dir_suffix(getattr(args, "datasets", "dentex,tufts"))
    prefix = _model_subdir_prefix(dir_suffix)
    cv_dir = Path(f"data/yolo_{dir_suffix}_cv")
    summary_path = cv_dir / "fold_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot find {summary_path}. Run: python scripts/prepare_yolo_dataset.py "
            f"--mode cv --datasets {getattr(args, 'datasets', 'dentex,tufts')}"
        )

    summary = json.loads(summary_path.read_text())
    n_folds = summary["n_folds"]
    resume = getattr(args, "resume", False)

    target_fold_arg = getattr(args, "target_fold", "all")
    if target_fold_arg is not None and str(target_fold_arg).lower() != "all":
        folds_to_run = [int(target_fold_arg)]
        print(f"Targeting single fold: {folds_to_run[0]} (out of {n_folds} total folds)")
    else:
        folds_to_run = list(range(n_folds))

    if resume:
        _ensure_folds_hydrated(model_root, prefix, n_folds)

    import torch
    device = getattr(args, "device", "0")
    if device in ("0", "cuda") and not torch.cuda.is_available():
        device = "cpu"

    results = []

    for fold in folds_to_run:
        yaml_path = cv_dir / f"fold_{fold}" / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing {yaml_path}")

        fold_dir = model_root / f"{prefix}cv_fold_{fold}"
        best_pt = fold_dir / "weights" / "best.pt"
        last_pt = fold_dir / "weights" / "last.pt"

        # A fold is complete only if YOLO finished and generated the final validation images
        confusion_matrix = fold_dir / "confusion_matrix.png"
        if resume and confusion_matrix.exists() and best_pt.exists():
            print(f"\n  FOLD {fold + 1}/{n_folds} — already complete, skipping.")
            
            csv_path = fold_dir / "results.csv"
            if csv_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    last_row = df.iloc[-1]
                    metrics = {
                        "fold": fold,
                        "map50": float(last_row.get("metrics/mAP50(B)", 0)),
                        "map50_95": float(last_row.get("metrics/mAP50-95(B)", 0)),
                        "precision": float(last_row.get("metrics/precision(B)", 0)),
                        "recall": float(last_row.get("metrics/recall(B)", 0)),
                    }
                    results.append(metrics)
                except Exception:
                    pass
            continue

        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        if resume and last_pt.exists():
            print(f"Resuming fold {fold} from checkpoint: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            # Clear stale label caches so YOLO re-indexes all dataset images
            dataset_dir = yaml_path.parent
            for cache_file in list(dataset_dir.glob("labels/**/*.cache")) + list(dataset_dir.glob("labels/*.cache")):
                try:
                    cache_file.unlink(missing_ok=True)
                except Exception:
                    pass
            model = YOLO(args.model)

        train_kwargs = dict(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(model_root),
            name=f"{prefix}cv_fold_{fold}",
            exist_ok=True,
        )
        if hasattr(args, "save_period") and args.save_period:
            train_kwargs["save_period"] = args.save_period
        if hasattr(args, "patience") and args.patience:
            train_kwargs["patience"] = args.patience

        result = model.train(**train_kwargs)

        # Defensively extract metrics from results_dict, box attributes, or results.csv
        metrics = {"fold": fold, "map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}
        if hasattr(result, "results_dict") and isinstance(result.results_dict, dict) and result.results_dict:
            res_d = result.results_dict
            metrics["map50"] = float(res_d.get("metrics/mAP50(B)", res_d.get("metrics/mAP50", 0.0)))
            metrics["map50_95"] = float(res_d.get("metrics/mAP50-95(B)", res_d.get("metrics/mAP50-95", 0.0)))
            metrics["precision"] = float(res_d.get("metrics/precision(B)", res_d.get("metrics/precision", 0.0)))
            metrics["recall"] = float(res_d.get("metrics/recall(B)", res_d.get("metrics/recall", 0.0)))
        elif hasattr(result, "box"):
            metrics["map50"] = float(getattr(result.box, "map50", 0.0))
            metrics["map50_95"] = float(getattr(result.box, "map", 0.0))
            metrics["precision"] = float(getattr(result.box, "mp", 0.0))
            metrics["recall"] = float(getattr(result.box, "mr", 0.0))

        csv_path = fold_dir / "results.csv"
        if csv_path.exists() and (metrics["map50"] == 0.0 and metrics["map50_95"] == 0.0):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                last_row = df.iloc[-1]
                metrics["map50"] = float(last_row.get("metrics/mAP50(B)", 0.0))
                metrics["map50_95"] = float(last_row.get("metrics/mAP50-95(B)", 0.0))
                metrics["precision"] = float(last_row.get("metrics/precision(B)", 0.0))
                metrics["recall"] = float(last_row.get("metrics/recall(B)", 0.0))
            except Exception:
                pass

        results.append(metrics)
        print(f"\nFold {fold}: mAP50={metrics['map50']:.4f}  mAP50-95={metrics['map50_95']:.4f}")

        # Clean up intermediate epoch checkpoints (keep best.pt and last.pt)
        weights_dir = fold_dir / "weights"
        if weights_dir.exists():
            for ep_file in weights_dir.glob("epoch*.pt"):
                try:
                    ep_file.unlink()
                except Exception:
                    pass

        # Sync completed fold to Hugging Face if configured
        if not getattr(args, "no_hf_sync", False):
            hf_token = os.environ.get("HF_TOKEN")
            hf_repo = os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models")
            if hf_token and not hf_token.startswith("YOUR_"):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
                    api.upload_folder(
                        folder_path=str(fold_dir),
                        path_in_repo=f"yolo_cv/{prefix}cv_fold_{fold}",
                        repo_id=hf_repo,
                        repo_type="model",
                        commit_message=f"Auto-sync YOLO CV fold {fold} complete",
                    )
                    print(f"  [HF Sync] Uploaded fold {fold} artifacts to {hf_repo}/yolo_cv/{prefix}cv_fold_{fold}")
                except Exception as e:
                    print(f"  [HF Sync] Notice: Could not sync fold {fold} to HF: {e}")

        # Free GPU memory before next fold
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # If running a single fold, stop after that fold completes
    if len(folds_to_run) == 1 and folds_to_run != list(range(n_folds)):
        print(f"\nFold {folds_to_run[0]} completed successfully!")
        return

    # --- Find best fold by mAP50-95 ---
    if not results:
        print("No fold results recorded.")
        return

    best = max(results, key=lambda x: x["map50_95"])
    best_fold = best["fold"]

    # --- Copy best fold weights to final location ---
    best_src = model_root / f"{prefix}cv_fold_{best_fold}" / "weights" / "best.pt"
    best_dst = model_root / f"{prefix}grounding_tool_cv_best" / "weights" / "best.pt"
    best_dst.parent.mkdir(parents=True, exist_ok=True)
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
    
    # Copy all training results (PR curves, confusion matrix, args.yaml, results.csv, etc.)
    best_fold_dir = model_root / f"{prefix}cv_fold_{best_fold}"
    if best_fold_dir.exists():
        for file_path in best_fold_dir.iterdir():
            if file_path.is_file():
                shutil.copy2(file_path, model_root / f"{prefix}grounding_tool_cv_best" / file_path.name)

    # Keep every fold's best model reachable for later ensemble/ablation work.
    for fold in range(n_folds):
        src = model_root / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt"
        if src.exists():
            ensemble_dst = model_root / f"{prefix}fold_best_models" / f"fold_{fold}_best.pt"
            ensemble_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, ensemble_dst)

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print("  CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  {'Fold':<6} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print(f"  {'-' * 46}")
    for r in results:
        marker = " <-- BEST" if r["fold"] == best_fold else ""
        print(
            f"  {r['fold']:<6} {r['map50']:<10.4f} {r['map50_95']:<10.4f} "
            f"{r['precision']:<10.4f} {r['recall']:<10.4f}{marker}"
        )
    print(f"  {'-' * 46}")

    mean_map50 = sum(r["map50"] for r in results) / len(results)
    std_map50 = (sum((r["map50"] - mean_map50) ** 2 for r in results) / len(results)) ** 0.5
    mean_map50_95 = sum(r["map50_95"] for r in results) / len(results)
    std_map50_95 = (sum((r["map50_95"] - mean_map50_95) ** 2 for r in results) / len(results)) ** 0.5

    print(f"  Mean mAP50:      {mean_map50:.4f} +/- {std_map50:.4f}")
    print(f"  Mean mAP50-95:   {mean_map50_95:.4f} +/- {std_map50_95:.4f}")
    print(f"\n  Best fold: {best_fold} (mAP50-95={best['map50_95']:.4f})")
    print(f"  Best weights: {best_dst}")

    # --- Save results JSON ---
    results_data = {
        "folds": results,
        "best_fold": best_fold,
        "mean_map50": mean_map50,
        "std_map50": std_map50,
        "mean_map50_95": mean_map50_95,
        "std_map50_95": std_map50_95,
    }
    results_path = model_root / f"{prefix}grounding_tool_cv_best" / "cv_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"  Results saved to: {results_path}")

    # Final sync to Hugging Face
    if not getattr(args, "no_hf_sync", False):
        hf_token = os.environ.get("HF_TOKEN")
        hf_repo = os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models")
        if hf_token and not hf_token.startswith("YOUR_"):
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
                api.upload_folder(
                    folder_path=str(model_root),
                    path_in_repo="yolo_cv",
                    repo_id=hf_repo,
                    repo_type="model",
                    commit_message="Final YOLO CV results & best model weights",
                )
                print(f"  [HF Sync] Successfully uploaded final CV results and best models to {hf_repo}/yolo_cv")
            except Exception as e:
                print(f"  [HF Sync] Notice: Could not push final models to HF: {e}")


def _ensure_folds_hydrated(model_root: Path, prefix: str, n_folds: int) -> list[int]:
    """Ensures all fold checkpoints are located and mapped into model_root / f'{prefix}cv_fold_{fold}'."""
    # 1. Check if all required fold checkpoints already exist locally on disk across common candidate directories
    missing = []
    for fold in range(n_folds):
        target = model_root / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt"
        if not target.exists():
            candidates = [
                model_root / f"dentex_cv_fold_{fold}" / "weights" / "best.pt",
                model_root / f"cv_fold_{fold}" / "weights" / "best.pt",
                model_root / f"dentex_tufts_cv_fold_{fold}" / "weights" / "best.pt",
                model_root / "fold_best_models" / f"fold_{fold}_best.pt",
                model_root / "dentex_tufts_fold_best_models" / f"fold_{fold}_best.pt",
                model_root / "dentex_fold_best_models" / f"fold_{fold}_best.pt",
                Path("data/yolo_cv") / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt",
                Path("data/yolo_cv") / f"cv_fold_{fold}" / "weights" / "best.pt",
                model_root / "yolo_cv" / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt",
                model_root / "yolo_cv" / f"dentex_cv_fold_{fold}" / "weights" / "best.pt",
                model_root / "yolo_cv" / f"dentex_tufts_cv_fold_{fold}" / "weights" / "best.pt",
            ]
            for cand in candidates:
                if cand.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cand, target)
                    break
        if not target.exists():
            missing.append(fold)

    if not missing:
        found_folds = []
        for fold in range(n_folds):
            weight_path = model_root / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt"
            if weight_path.exists():
                found_folds.append(fold)
        return found_folds

    # 2. Download missing fold checkpoints from HF Hub only if files are genuinely missing
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models")
    if hf_token and not hf_token.startswith("YOUR_"):
        try:
            from huggingface_hub import snapshot_download
            print(f"Checking Hugging Face Hub ({hf_repo}/yolo_cv) for {len(missing)} missing fold checkpoints...")
            staging_dir = model_root / "_hf_staging"
            snapshot_download(
                repo_id=hf_repo,
                repo_type="model",
                allow_patterns=["yolo_cv/**"],
                local_dir=str(staging_dir),
                token=hf_token,
            )
            yolo_cv_staged = staging_dir / "yolo_cv"
            if yolo_cv_staged.exists():
                for item in yolo_cv_staged.iterdir():
                    dest = model_root / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest)
            shutil.rmtree(staging_dir, ignore_errors=True)
        except Exception as e:
            print(f"HF auto-hydration notice: {e}")

    found_folds = []
    for fold in range(n_folds):
        weight_path = model_root / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt"
        if weight_path.exists():
            found_folds.append(fold)
    return found_folds


def _compute_box_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection-over-Union between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_target_grounding(
    model: YOLO,
    val_images_df: pd.DataFrame,
    val_annots_df: pd.DataFrame,
    conf_thresh: float = 0.001,
    nominal_conf_thresh: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate YOLO model specifically against the annotated target teeth present in ground truth.
    
    Uses greedy 1-to-1 bipartite matching to avoid double-counting true positives and avoids
    penalizing unannotated healthy teeth on partial/disease benchmark splits.
    
    Collects detections down to conf_thresh=0.001 for complete 101-point COCO PR curve integration (mAP50),
    while computing nominal operating point metrics (Recall, Precision, Mean IoU) at nominal_conf_thresh=0.25.
    """
    from dental_agent.data.dentex import dentex_row_to_fdi

    # Ensure device validity
    import torch
    if device in ("0", "cuda") and not torch.cuda.is_available():
        device = "cpu"

    total_targets = 0
    tp_50_nominal = 0
    tp_75_nominal = 0
    target_pred_nominal_count = 0
    target_tp_nominal_count = 0
    target_nominal_ious = []
    
    # Standard 10 COCO IoU thresholds (0.50 to 0.95)
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    
    # Store predictions and ground truths for precision-recall curve computation
    # per target class: {class_idx: {"preds": [list of {"conf": float, "tp_by_iou": list}], "n_gt": int}}
    class_eval_records = {c: {"preds": [], "n_gt": 0} for c in range(32)}

    for _, img_row in val_images_df.iterrows():
        img_id = img_row["id"]
        local_path = img_row.get("local_path")
        if not local_path or not Path(str(local_path)).exists():
            continue

        img_w = float(img_row["width"])
        img_h = float(img_row["height"])

        if val_annots_df.empty or "image_id" not in val_annots_df.columns:
            continue

        img_annots = val_annots_df[val_annots_df["image_id"] == img_id]
        if img_annots.empty:
            continue

        # Extract Ground Truth targets for this image
        gt_targets = []
        gt_classes_in_img = set()
        for _, ann in img_annots.iterrows():
            if pd.isna(ann.get("category_id_1")) or pd.isna(ann.get("category_id_2")):
                continue
            quadrant, position = dentex_row_to_fdi(ann)
            if not (1 <= quadrant <= 4 and 1 <= position <= 8):
                continue
            cls_idx = (quadrant - 1) * 8 + (position - 1)
            bbox = ann.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x_min, y_min, bw, bh = bbox
            x1, y1, x2, y2 = x_min, y_min, x_min + bw, y_min + bh
            # Safe clamping
            x1 = max(0.0, min(img_w, x1))
            y1 = max(0.0, min(img_h, y1))
            x2 = max(0.0, min(img_w, x2))
            y2 = max(0.0, min(img_h, y2))
            gt_targets.append((cls_idx, [x1, y1, x2, y2]))
            gt_classes_in_img.add(cls_idx)
            class_eval_records[cls_idx]["n_gt"] += 1

        if not gt_targets:
            continue

        total_targets += len(gt_targets)

        # Run model inference on this test image at low confidence to capture full PR curve
        try:
            from PIL import Image
            img_input = str(local_path)
            try:
                with Image.open(str(local_path)) as im:
                    img_input = im.convert("RGB")
            except Exception:
                img_input = str(local_path)

            preds = model.predict(source=img_input, conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
        except Exception as e:
            print(f"Inference notice on {local_path}: {e}")
            continue

        pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.zeros((0, 4))
        pred_classes = preds.boxes.cls.cpu().numpy().astype(int) if len(preds.boxes) else np.zeros((0,), dtype=int)
        pred_confs = preds.boxes.conf.cpu().numpy() if len(preds.boxes) else np.zeros((0,))

        # 1. Greedy 1-to-1 matching for nominal operating point (conf >= nominal_conf_thresh)
        nominal_mask = pred_confs >= nominal_conf_thresh
        nom_boxes = pred_boxes[nominal_mask]
        nom_classes = pred_classes[nominal_mask]
        nom_confs = pred_confs[nominal_mask]

        nom_order = np.argsort(-nom_confs) if len(nom_confs) else []
        matched_nom_gt_indices = set()
        matched_gt_best_ious = {}

        for idx in nom_order:
            p_cls = nom_classes[idx]
            if p_cls in gt_classes_in_img:
                target_pred_nominal_count += 1
                p_box = nom_boxes[idx].tolist()
                matching_gts = [
                    (g_i, gb) for g_i, (c, gb) in enumerate(gt_targets)
                    if c == p_cls and g_i not in matched_nom_gt_indices
                ]
                if matching_gts:
                    best_g_i, best_gb = max(matching_gts, key=lambda item: _compute_box_iou(p_box, item[1]))
                    iou = _compute_box_iou(p_box, best_gb)
                    if iou >= 0.50:
                        target_tp_nominal_count += 1
                        matched_nom_gt_indices.add(best_g_i)
                        matched_gt_best_ious[best_g_i] = iou

        for g_i, (gt_cls, gt_box) in enumerate(gt_targets):
            if g_i in matched_nom_gt_indices:
                iou_val = matched_gt_best_ious[g_i]
                target_nominal_ious.append(iou_val)
                tp_50_nominal += 1
                if iou_val >= 0.75:
                    tp_75_nominal += 1
            else:
                cand_nom_idx = [i for i, c in enumerate(nom_classes) if c == gt_cls]
                best_sub_iou = max([_compute_box_iou(nom_boxes[i].tolist(), gt_box) for i in cand_nom_idx], default=0.0)
                target_nominal_ious.append(best_sub_iou)

        # 2. Record all target predictions for continuous PR curve calculation across 10 COCO IoU thresholds (0.50 to 0.95)
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        pred_order = np.argsort(-pred_confs) if len(pred_confs) else []
        matched_gt_per_iou = [set() for _ in range(len(iou_thresholds))]

        for idx in pred_order:
            p_cls = int(pred_classes[idx])
            if p_cls in gt_classes_in_img:
                p_box = pred_boxes[idx].tolist()
                p_conf = float(pred_confs[idx])

                tp_flags = []
                for t_idx, iou_t in enumerate(iou_thresholds):
                    matching_gts = [
                        (g_i, gb)
                        for g_i, (c, gb) in enumerate(gt_targets)
                        if c == p_cls and g_i not in matched_gt_per_iou[t_idx]
                    ]
                    is_tp = 0.0
                    if matching_gts:
                        best_g_i, best_gb = max(matching_gts, key=lambda item: _compute_box_iou(p_box, item[1]))
                        if _compute_box_iou(p_box, best_gb) >= iou_t:
                            is_tp = 1.0
                            matched_gt_per_iou[t_idx].add(best_g_i)
                    tp_flags.append(is_tp)

                class_eval_records[p_cls]["preds"].append({"conf": p_conf, "tp_by_iou": tp_flags})

    # Calculate nominal operating metrics
    recall_50 = (tp_50_nominal / max(1, total_targets))
    recall_75 = (tp_75_nominal / max(1, total_targets))
    precision = (target_tp_nominal_count / max(1, target_pred_nominal_count)) if target_pred_nominal_count > 0 else 0.0
    mean_iou = (sum(target_nominal_ious) / max(1, len(target_nominal_ious))) if target_nominal_ious else 0.0

    # Compute target class mAP across all 10 COCO IoU thresholds (0.50 to 0.95) using 101-point continuous interpolation
    aps_per_iou = [[] for _ in range(len(iou_thresholds))]
    for c, rec in class_eval_records.items():
        if rec["n_gt"] == 0:
            continue
        if not rec["preds"]:
            for t_idx in range(len(iou_thresholds)):
                aps_per_iou[t_idx].append(0.0)
            continue

        preds_list = rec["preds"]
        preds_list.sort(key=lambda x: x["conf"], reverse=True)

        for t_idx in range(len(iou_thresholds)):
            tps = [x["tp_by_iou"][t_idx] for x in preds_list]
            cumsum_tp = np.cumsum(tps)
            cumsum_fp = np.cumsum([1.0 - x for x in tps])
            precisions = cumsum_tp / np.maximum(1e-8, cumsum_tp + cumsum_fp)
            recalls = cumsum_tp / float(rec["n_gt"])

            ap = 0.0
            for t in np.linspace(0, 1, 101):
                p_over = precisions[recalls >= t]
                p = np.max(p_over) if len(p_over) > 0 else 0.0
                ap += p / 101.0
            aps_per_iou[t_idx].append(float(ap))

    map50 = float(np.mean(aps_per_iou[0])) if aps_per_iou[0] else recall_50
    mean_maps_across_iou = [float(np.mean(aps)) if aps else 0.0 for aps in aps_per_iou]
    map50_95 = float(np.mean(mean_maps_across_iou)) if mean_maps_across_iou else map50

    return {
        "recall_50": recall_50,
        "recall_75": recall_75,
        "precision": precision,
        "mean_iou": mean_iou,
        "map50": map50,
        "map50_95": map50_95,
        "total_targets": total_targets,
    }


def evaluate_yolo_labels_target_grounding(
    model: YOLO,
    img_dir: Path | str,
    label_dir: Path | str,
    conf_thresh: float = 0.001,
    nominal_conf_thresh: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate YOLO model against in-fold validation splits stored in YOLO .txt format.
    
    Loads (cls_idx, xc, yc, w, h) from label_dir/*.txt, matches against model predictions
    using greedy 1-to-1 bipartite matching, and evaluates continuous 10-threshold 101-point COCO PR curve integration.
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    if not img_dir.exists() or not label_dir.exists():
        return {}

    img_exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    img_files = sorted([p for p in img_dir.iterdir() if p.suffix in img_exts])
    if not img_files:
        return {}

    from PIL import Image
    import torch
    if device in ("0", "cuda") and not torch.cuda.is_available():
        device = "cpu"

    total_targets = 0
    tp_50_nominal = 0
    tp_75_nominal = 0
    target_pred_nominal_count = 0
    target_tp_nominal_count = 0
    target_nominal_ious = []
    
    # Standard 10 COCO IoU thresholds (0.50 to 0.95)
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    
    class_eval_records = {c: {"preds": [], "n_gt": 0} for c in range(32)}

    for img_p in img_files:
        lbl_p = label_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        try:
            with Image.open(img_p) as im:
                img_w, img_h = float(im.width), float(im.height)
                img_rgb = im.convert("RGB")
        except Exception:
            continue

        gt_targets = []
        gt_classes_in_img = set()
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    c_idx = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                except ValueError:
                    continue
                if not (0 <= c_idx < 32):
                    continue
                x1 = max(0.0, min(img_w, (xc - w / 2.0) * img_w))
                y1 = max(0.0, min(img_h, (yc - h / 2.0) * img_h))
                x2 = max(0.0, min(img_w, (xc + w / 2.0) * img_w))
                y2 = max(0.0, min(img_h, (yc + h / 2.0) * img_h))
                gt_targets.append((c_idx, [x1, y1, x2, y2]))
                gt_classes_in_img.add(c_idx)
                class_eval_records[c_idx]["n_gt"] += 1

        if not gt_targets:
            continue

        total_targets += len(gt_targets)

        try:
            preds = model.predict(source=img_rgb, conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
        except Exception as e:
            continue

        pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.zeros((0, 4))
        pred_classes = preds.boxes.cls.cpu().numpy().astype(int) if len(preds.boxes) else np.zeros((0,), dtype=int)
        pred_confs = preds.boxes.conf.cpu().numpy() if len(preds.boxes) else np.zeros((0,))

        # 1. Greedy 1-to-1 matching for nominal operating point (conf >= nominal_conf_thresh)
        nominal_mask = pred_confs >= nominal_conf_thresh
        nom_boxes = pred_boxes[nominal_mask]
        nom_classes = pred_classes[nominal_mask]
        nom_confs = pred_confs[nominal_mask]

        nom_order = np.argsort(-nom_confs) if len(nom_confs) else []
        matched_nom_gt_indices = set()
        matched_gt_best_ious = {}

        for idx in nom_order:
            p_cls = nom_classes[idx]
            if p_cls in gt_classes_in_img:
                target_pred_nominal_count += 1
                p_box = nom_boxes[idx].tolist()
                matching_gts = [
                    (g_i, gb) for g_i, (c, gb) in enumerate(gt_targets)
                    if c == p_cls and g_i not in matched_nom_gt_indices
                ]
                if matching_gts:
                    best_g_i, best_gb = max(matching_gts, key=lambda item: _compute_box_iou(p_box, item[1]))
                    iou = _compute_box_iou(p_box, best_gb)
                    if iou >= 0.50:
                        target_tp_nominal_count += 1
                        matched_nom_gt_indices.add(best_g_i)
                        matched_gt_best_ious[best_g_i] = iou

        for g_i, (gt_cls, gt_box) in enumerate(gt_targets):
            if g_i in matched_nom_gt_indices:
                iou_val = matched_gt_best_ious[g_i]
                target_nominal_ious.append(iou_val)
                tp_50_nominal += 1
                if iou_val >= 0.75:
                    tp_75_nominal += 1
            else:
                cand_nom_idx = [i for i, c in enumerate(nom_classes) if c == gt_cls]
                best_sub_iou = max([_compute_box_iou(nom_boxes[i].tolist(), gt_box) for i in cand_nom_idx], default=0.0)
                target_nominal_ious.append(best_sub_iou)

        # 2. Record all target predictions for continuous PR curve calculation across 10 COCO IoU thresholds (0.50 to 0.95)
        iou_thresholds = np.linspace(0.50, 0.95, 10)
        pred_order = np.argsort(-pred_confs) if len(pred_confs) else []
        matched_gt_per_iou = [set() for _ in range(len(iou_thresholds))]

        for idx in pred_order:
            p_cls = int(pred_classes[idx])
            if p_cls in gt_classes_in_img:
                p_box = pred_boxes[idx].tolist()
                p_conf = float(pred_confs[idx])

                tp_flags = []
                for t_idx, iou_t in enumerate(iou_thresholds):
                    matching_gts = [
                        (g_i, gb)
                        for g_i, (c, gb) in enumerate(gt_targets)
                        if c == p_cls and g_i not in matched_gt_per_iou[t_idx]
                    ]
                    is_tp = 0.0
                    if matching_gts:
                        best_g_i, best_gb = max(matching_gts, key=lambda item: _compute_box_iou(p_box, item[1]))
                        if _compute_box_iou(p_box, best_gb) >= iou_t:
                            is_tp = 1.0
                            matched_gt_per_iou[t_idx].add(best_g_i)
                    tp_flags.append(is_tp)

                class_eval_records[p_cls]["preds"].append({"conf": p_conf, "tp_by_iou": tp_flags})

    recall_50 = (tp_50_nominal / max(1, total_targets))
    recall_75 = (tp_75_nominal / max(1, total_targets))
    precision = (target_tp_nominal_count / max(1, target_pred_nominal_count)) if target_pred_nominal_count > 0 else 0.0
    mean_iou = (sum(target_nominal_ious) / max(1, len(target_nominal_ious))) if target_nominal_ious else 0.0

    aps_per_iou = [[] for _ in range(len(iou_thresholds))]
    for c, rec in class_eval_records.items():
        if rec["n_gt"] == 0:
            continue
        if not rec["preds"]:
            for t_idx in range(len(iou_thresholds)):
                aps_per_iou[t_idx].append(0.0)
            continue
        preds_list = rec["preds"]
        preds_list.sort(key=lambda x: x["conf"], reverse=True)

        for t_idx in range(len(iou_thresholds)):
            tps = [x["tp_by_iou"][t_idx] for x in preds_list]
            cumsum_tp = np.cumsum(tps)
            cumsum_fp = np.cumsum([1.0 - x for x in tps])
            precisions = cumsum_tp / np.maximum(1e-8, cumsum_tp + cumsum_fp)
            recalls = cumsum_tp / float(rec["n_gt"])
            ap = 0.0
            for t in np.linspace(0, 1, 101):
                p_over = precisions[recalls >= t]
                p = np.max(p_over) if len(p_over) > 0 else 0.0
                ap += p / 101.0
            aps_per_iou[t_idx].append(float(ap))

    map50 = float(np.mean(aps_per_iou[0])) if aps_per_iou[0] else recall_50
    mean_maps_across_iou = [float(np.mean(aps)) if aps else 0.0 for aps in aps_per_iou]
    map50_95 = float(np.mean(mean_maps_across_iou)) if mean_maps_across_iou else map50

    return {
        "recall_50": recall_50,
        "recall_75": recall_75,
        "precision": precision,
        "mean_iou": mean_iou,
        "map50": map50,
        "map50_95": map50_95,
        "total_targets": total_targets,
    }


def evaluate_benchmark(args):
    """Evaluate all trained fold models across both 5-fold CV splits and the held-out test set."""
    model_root = get_model_root()
    dir_suffix = _dataset_dir_suffix(getattr(args, "datasets", "dentex,tufts"))
    prefix = _model_subdir_prefix(dir_suffix)

    # Load authoritative held-out test set (validation split from DENTEX)
    val_images_df = None
    val_annots_df = None
    try:
        from dental_agent.data.dentex import load_dentex_dataset
        from scripts.prepare_yolo_dataset import _ensure_images_downloaded
        data_dir = getattr(args, "data_dir", None)
        val_images_df, val_annots_df, _ = load_dentex_dataset(
            data_dir=data_dir,
            split_name="validation",
            combine_enumeration_splits=False,
        )
        val_images_df = _ensure_images_downloaded(val_images_df, "dentex", data_dir=data_dir)
        val_images_df = val_images_df[val_images_df["local_path"].notna()].copy()
    except Exception as e:
        print(f"  [Benchmark] Notice: Could not load raw validation DataFrame: {e}")

    # Fallback: check local YOLO test directories on disk if DataFrame could not be resolved
    test_img_dir = None
    test_lbl_dir = None
    if val_images_df is None or len(val_images_df) == 0:
        for cand_root in [
            Path(f"data/yolo_{dir_suffix}_cv"),
            Path("data/yolo_dentex_tufts_cv"),
            Path("data/yolo_dentex_cv"),
            Path(f"data/yolo_{dir_suffix}"),
            Path("data/yolo_dentex"),
            Path("data/yolo_cv"),
        ]:
            for img_sub, lbl_sub in [
                (cand_root / "test" / "images", cand_root / "test" / "labels"),
                (cand_root / "images" / "test", cand_root / "labels" / "test"),
                (cand_root / "images" / "validation", cand_root / "labels" / "validation"),
            ]:
                if img_sub.exists() and lbl_sub.exists() and any(img_sub.iterdir()) and any(lbl_sub.iterdir()):
                    test_img_dir = img_sub
                    test_lbl_dir = lbl_sub
                    break
            if test_img_dir and test_lbl_dir:
                break

    # Hydrate both model families from local disk or Hugging Face Hub (checks disk first!)
    n_folds = 5
    _ensure_folds_hydrated(model_root, "dentex_tufts_", n_folds)
    _ensure_folds_hydrated(model_root, "dentex_", n_folds)

    device = str(getattr(args, "device", "0"))
    import torch
    if device in ("0", "cuda") and not torch.cuda.is_available():
        device = "cpu"

    # --- Part 1: In-Fold Cross-Validation Target Grounding Benchmark (Target-Filtered) ---
    cv_val_records = []
    cv_configs = [
        ("DENTEX-Only", ["dentex_cv_fold_", "cv_fold_"], ["dentex"]),
        ("DENTEX+Tufts", ["dentex_tufts_cv_fold_"], ["dentex", "tufts"]),
    ]

    all_cv_results = {}
    from sklearn.model_selection import KFold

    train_pools_cache = {}
    for family_name, prefix_options, dataset_names in cv_configs:
        family_cv_metrics = []
        d_suffix = _dataset_dir_suffix(",".join(dataset_names))

        # Check if local YOLO in-fold validation folders exist on disk first across all candidates
        local_yolo_splits = {}
        for fold in range(n_folds):
            for cv_root in [
                Path(f"data/yolo_{d_suffix}_cv"),
                Path("data/yolo_dentex_tufts_cv"),
                Path("data/yolo_dentex_cv"),
                Path(f"data/yolo_{d_suffix}"),
                Path("data/yolo_cv"),
            ]:
                cand_img = cv_root / f"fold_{fold}" / "images" / "val"
                cand_lbl = cv_root / f"fold_{fold}" / "labels" / "val"
                if cand_img.exists() and cand_lbl.exists() and any(cand_img.iterdir()):
                    local_yolo_splits[fold] = (cand_img, cand_lbl)
                    break

        train_pools = []
        combined_keys = []
        fold_splits = []
        if len(local_yolo_splits) < n_folds:
            # Lazy fallback: load raw dataset pools only if local YOLO directories are missing
            for dname in dataset_names:
                if dname in train_pools_cache:
                    train_pools.append(train_pools_cache[dname])
                    continue
                if dname == "dentex":
                    from dental_agent.data.dentex import load_dentex_dataset
                    from scripts.prepare_yolo_dataset import _ensure_images_downloaded
                    data_dir = getattr(args, "data_dir", None)
                    imgs_df, ann_df, _ = load_dentex_dataset(data_dir=data_dir, split_name="train", combine_enumeration_splits=True)
                    imgs_df = _ensure_images_downloaded(imgs_df, "dentex", data_dir=data_dir)
                    imgs_df = imgs_df[imgs_df["local_path"].notna()].copy()
                    train_pools_cache[dname] = (dname, imgs_df, ann_df)
                    train_pools.append((dname, imgs_df, ann_df))
                elif dname == "tufts":
                    from dental_agent.data.tufts import load_tufts_tooth_boxes
                    from scripts.prepare_yolo_dataset import _ensure_images_downloaded
                    data_dir = getattr(args, "data_dir", None)
                    try:
                        imgs_df, ann_df, _ = load_tufts_tooth_boxes(data_dir=data_dir)
                        imgs_df = _ensure_images_downloaded(imgs_df, "tufts", data_dir=data_dir)
                        imgs_df = imgs_df[imgs_df["local_path"].notna()].copy()
                        train_pools_cache[dname] = (dname, imgs_df, ann_df)
                        train_pools.append((dname, imgs_df, ann_df))
                    except Exception:
                        pass

            combined_keys = [
                (dname, img_id)
                for dname, images_df, _ in train_pools
                for img_id in sorted(images_df["id"].unique())
            ]
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_splits = list(kf.split(combined_keys)) if combined_keys else []

        for fold in range(n_folds):
            weight_path = None
            for pref in prefix_options:
                cand = model_root / f"{pref}{fold}" / "weights" / "best.pt"
                if cand.exists():
                    weight_path = cand
                    break
                alt_prefixed = model_root / f"{pref.rstrip('_')}_fold_best_models" / f"fold_{fold}_best.pt"
                if alt_prefixed.exists():
                    weight_path = alt_prefixed
                    break
                if family_name == "DENTEX+Tufts" or pref.startswith("dentex_tufts"):
                    alt_default = model_root / "fold_best_models" / f"fold_{fold}_best.pt"
                    if alt_default.exists():
                        weight_path = alt_default
                        break

            if not weight_path or not weight_path.exists():
                continue

            fold_img_dir, fold_lbl_dir = local_yolo_splits.get(fold, (None, None))

            print(f"  Evaluating {family_name} Fold {fold + 1}/{n_folds} ({weight_path.name})...")
            model = YOLO(str(weight_path))
            res = None
            if fold_img_dir and fold_lbl_dir:
                res = evaluate_yolo_labels_target_grounding(
                    model, fold_img_dir, fold_lbl_dir, conf_thresh=0.001, nominal_conf_thresh=0.25, imgsz=args.imgsz, device=device
                )
            elif fold_splits and fold < len(fold_splits):
                _, val_idx = fold_splits[fold]
                val_keys = {combined_keys[i] for i in val_idx}
                
                fold_val_images = []
                fold_val_annots = []
                for dname, images_df, ann_df in train_pools:
                    fold_val_ids = {img_id for (dn, img_id) in val_keys if dn == dname}
                    v_imgs = images_df[images_df["id"].isin(fold_val_ids)].copy()
                    v_anns = ann_df[ann_df["image_id"].isin(fold_val_ids)].copy()
                    fold_val_images.append(v_imgs)
                    fold_val_annots.append(v_anns)
                
                if fold_val_images:
                    merged_imgs = pd.concat(fold_val_images, ignore_index=True)
                    merged_anns = pd.concat(fold_val_annots, ignore_index=True)
                    res = evaluate_target_grounding(
                        model, merged_imgs, merged_anns, conf_thresh=0.001, nominal_conf_thresh=0.25, imgsz=args.imgsz, device=device
                    )

            if not res:
                continue

            entry = {
                "family": family_name,
                "name": f"{family_name} (CV Fold {fold})",
                "fold": fold,
                "weight_path": str(weight_path),
                **res,
            }
            family_cv_metrics.append(entry)
            cv_val_records.append(entry)
            print(
                f"  -> {entry['name']:<30} mAP50={entry['map50']:.4f} mAP50-95={entry['map50_95']:.4f} "
                f"Precision={entry['precision']:.4f} Rec@0.50={entry['recall_50']:.4f} Mean IoU={entry['mean_iou']:.4f}"
            )

        if family_cv_metrics:
            all_cv_results[family_name] = family_cv_metrics
            mean_rec50 = sum(m["recall_50"] for m in family_cv_metrics) / len(family_cv_metrics)
            mean_rec75 = sum(m["recall_75"] for m in family_cv_metrics) / len(family_cv_metrics)
            mean_prec = sum(m["precision"] for m in family_cv_metrics) / len(family_cv_metrics)
            mean_iou_val = sum(m["mean_iou"] for m in family_cv_metrics) / len(family_cv_metrics)
            mean_map = sum(m["map50"] for m in family_cv_metrics) / len(family_cv_metrics)
            mean_map50_95 = sum(m["map50_95"] for m in family_cv_metrics) / len(family_cv_metrics)
            summary_entry = {
                "family": family_name,
                "name": f"{family_name} (5-Fold Mean)",
                "fold": "mean",
                "recall_50": mean_rec50,
                "recall_75": mean_rec75,
                "precision": mean_prec,
                "mean_iou": mean_iou_val,
                "map50": mean_map,
                "map50_95": mean_map50_95,
            }
            cv_val_records.append(summary_entry)

            pref = "dentex_tufts_" if family_name == "DENTEX+Tufts" else ("dentex_" if family_name == "DENTEX-Only" else "")
            cv_out = model_root / f"{pref}grounding_tool_cv_best" / "cv_results.json"
            cv_out.parent.mkdir(parents=True, exist_ok=True)
            cv_data = {
                "family": family_name,
                "folds": family_cv_metrics,
                "mean_recall_50": mean_rec50,
                "mean_recall_75": mean_rec75,
                "mean_precision": mean_prec,
                "mean_iou": mean_iou_val,
                "mean_map50": mean_map,
                "mean_map50_95": mean_map50_95,
            }
            cv_out.write_text(json.dumps(cv_data, indent=2))
            print(f"  Saved clean target-filtered CV results to: {cv_out}")

    if cv_val_records:
        print(f"\n{'=' * 110}")
        print(f"  IN-FOLD TARGET TOOTH GROUNDING BENCHMARK (CV Validation Splits - Target-Filtered)")
        print(f"{'=' * 110}")
        print(f"  {'Model Architecture / Fold':<30} {'mAP50':<11} {'mAP50-95':<11} {'Precision':<11} {'Rec@0.50':<11} {'Rec@0.75':<11} {'Mean IoU':<11}")
        print(f"  {'-' * 110}")
        for r in cv_val_records:
            if r.get("fold") == "mean":
                print(f"  {'-' * 110}")
            print(
                f"  {r['name']:<30} {r['map50']:<11.4f} {r['map50_95']:<11.4f} "
                f"{r['precision']:<11.4f} {r['recall_50']:<11.4f} {r['recall_75']:<11.4f} {r['mean_iou']:<11.4f}"
            )
            if r.get("fold") == "mean":
                print(f"  {'-' * 110}")
        cv_val_json = model_root / "cv_val_target_evaluation.json"
        cv_val_json.write_text(json.dumps(cv_val_records, indent=2))

    # --- Part 3: Head-to-Head Target Grounding Benchmark Table ---
    n_test_imgs = len(list(test_img_dir.iterdir())) if test_img_dir else (len(val_images_df) if val_images_df is not None else 0)
    comparison_records = []
    
    # Evaluate both families: DENTEX-Only and DENTEX+Tufts
    families = [
        ("DENTEX-Only", ["dentex_cv_fold_", "cv_fold_"]),
        ("DENTEX+Tufts", ["dentex_tufts_cv_fold_"]),
    ]

    for family_name, prefix_options in families:
        family_metrics = []
        for fold in range(n_folds):
            weight_path = None
            for pref in prefix_options:
                cand = model_root / f"{pref}{fold}" / "weights" / "best.pt"
                if cand.exists():
                    weight_path = cand
                    break
                alt_prefixed = model_root / f"{pref.rstrip('_')}_fold_best_models" / f"fold_{fold}_best.pt"
                if alt_prefixed.exists():
                    weight_path = alt_prefixed
                    break
                if family_name == "DENTEX+Tufts" or pref.startswith("dentex_tufts"):
                    alt_default = model_root / "fold_best_models" / f"fold_{fold}_best.pt"
                    if alt_default.exists():
                        weight_path = alt_default
                        break

            if not weight_path or not weight_path.exists():
                continue

            print(f"  Evaluating {family_name} Fold {fold + 1}/{n_folds} on held-out test set ({weight_path.name})...")
            model = YOLO(str(weight_path))
            if val_images_df is not None and val_annots_df is not None and len(val_images_df) > 0:
                res = evaluate_target_grounding(
                    model, val_images_df, val_annots_df, conf_thresh=0.001, nominal_conf_thresh=0.25, imgsz=args.imgsz, device=device
                )
            elif test_img_dir and test_lbl_dir:
                res = evaluate_yolo_labels_target_grounding(
                    model, test_img_dir, test_lbl_dir, conf_thresh=0.001, nominal_conf_thresh=0.25, imgsz=args.imgsz, device=device
                )
            else:
                continue

            entry = {
                "family": family_name,
                "name": f"{family_name} (Fold {fold})",
                "fold": fold,
                "weight_path": str(weight_path),
                **res,
            }
            family_metrics.append(entry)
            comparison_records.append(entry)
            print(
                f"  -> {entry['name']:<30} mAP50={entry['map50']:.4f} mAP50-95={entry['map50_95']:.4f} "
                f"Precision={entry['precision']:.4f} Rec@0.50={entry['recall_50']:.4f} Mean IoU={entry['mean_iou']:.4f}"
            )

        if family_metrics:
            mean_rec50 = sum(m["recall_50"] for m in family_metrics) / len(family_metrics)
            mean_rec75 = sum(m["recall_75"] for m in family_metrics) / len(family_metrics)
            mean_prec = sum(m["precision"] for m in family_metrics) / len(family_metrics)
            mean_iou_val = sum(m["mean_iou"] for m in family_metrics) / len(family_metrics)
            mean_map = sum(m["map50"] for m in family_metrics) / len(family_metrics)
            mean_map50_95 = sum(m["map50_95"] for m in family_metrics) / len(family_metrics)
            summary_entry = {
                "family": family_name,
                "name": f"{family_name} (5-Fold Mean)",
                "fold": "mean",
                "recall_50": mean_rec50,
                "recall_75": mean_rec75,
                "precision": mean_prec,
                "mean_iou": mean_iou_val,
                "map50": mean_map,
                "map50_95": mean_map50_95,
            }
            comparison_records.append(summary_entry)

    if comparison_records:
        print(f"\n{'=' * 110}")
        print(f"  HELD-OUT TARGET TOOTH GROUNDING BENCHMARK (Official DENTEX Test Set - {n_test_imgs} Images)")
        print(f"{'=' * 110}")
        print(f"  {'Model Architecture / Fold':<30} {'mAP50':<11} {'mAP50-95':<11} {'Precision':<11} {'Rec@0.50':<11} {'Rec@0.75':<11} {'Mean IoU':<11}")
        print(f"  {'-' * 110}")
        for r in comparison_records:
            if r.get("fold") == "mean":
                print(f"  {'-' * 110}")
            print(
                f"  {r['name']:<30} {r['map50']:<11.4f} {r['map50_95']:<11.4f} "
                f"{r['precision']:<11.4f} {r['recall_50']:<11.4f} {r['recall_75']:<11.4f} {r['mean_iou']:<11.4f}"
            )
            if r.get("fold") == "mean":
                print(f"  {'-' * 110}")

    # Save benchmark results & copy top performer weights to grounding_tool_cv_best
    if comparison_records:
        valid_folds = [r for r in comparison_records if isinstance(r.get("fold"), int)]
        if valid_folds:
            best_benchmark = max(valid_folds, key=lambda x: x["map50"])
            print(f"  Top Benchmark Performer: {best_benchmark['name']} (mAP50: {best_benchmark['map50']:.4f}, Mean IoU: {best_benchmark['mean_iou']:.4f})")

            win_src = Path(best_benchmark["weight_path"])
            best_dst = model_root / f"{prefix}grounding_tool_cv_best" / "weights" / "best.pt"
            best_dst.parent.mkdir(parents=True, exist_ok=True)
            if win_src.exists():
                shutil.copy2(win_src, best_dst)
                print(f"  Assigned top benchmark model ({best_benchmark['name']}) -> {best_dst}")

            # Keep every fold's best model in fold_best_models/
            for fold in range(n_folds):
                src = model_root / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt"
                if not src.exists():
                    src = model_root / f"dentex_tufts_cv_fold_{fold}" / "weights" / "best.pt"
                if src.exists():
                    ensemble_dst = model_root / f"{prefix}fold_best_models" / f"fold_{fold}_best.pt"
                    ensemble_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, ensemble_dst)

            out_json = model_root / f"{prefix}grounding_tool_cv_best" / "benchmark_evaluation.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(comparison_records, indent=2))
            
            comp_json = model_root / "benchmark_comparative_evaluation.json"
            comp_json.write_text(json.dumps(comparison_records, indent=2))
            print(f"  Benchmark comparative results saved to: {comp_json}")

        # Sync best models, visual artifacts, and benchmark results to Hugging Face
        if not getattr(args, "no_hf_sync", False):
            hf_token = os.environ.get("HF_TOKEN")
            hf_repo = os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models")
            if hf_token and not hf_token.startswith("YOUR_"):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi(token=hf_token)
                    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
                    
                    staging_dir = model_root / "_hf_staging"
                    shutil.rmtree(staging_dir, ignore_errors=True)

                    max_retries = 3
                    for attempt in range(1, max_retries + 1):
                        try:
                            api.upload_folder(
                                folder_path=str(model_root),
                                path_in_repo="yolo_cv",
                                repo_id=hf_repo,
                                repo_type="model",
                                allow_patterns=[
                                    "**/*.pt",
                                    "**/*.json",
                                    "**/*.csv",
                                    "**/*.yaml",
                                    "**/*.png",
                                    "**/*.jpg",
                                    "**/*.jpeg",
                                ],
                                ignore_patterns=[
                                    "**/_hf_staging/**",
                                    "**/__pycache__/**",
                                    "**/epoch*.pt",
                                ],
                                commit_message="Final YOLO 5-Fold CV results, visual artifacts & benchmark weights",
                            )
                            print(f"  [HF Sync] Successfully uploaded all fold checkpoints and results to {hf_repo}/yolo_cv")
                            break
                        except Exception as upload_err:
                            if attempt < max_retries:
                                print(f"  [HF Sync] Attempt {attempt}/{max_retries} failed ({upload_err}). Retrying in 5s...")
                                time.sleep(5)
                            else:
                                raise upload_err
                except Exception as e:
                    print(f"  [HF Sync] Notice: Could not upload benchmark results to HF: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Grounding Tool.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every X epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs without improvement, default 10)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="Base model (yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on (e.g., '0' for GPU 0, 'cpu' for CPU)")
    parser.add_argument("--cross-validate", action="store_true", help="Run cross-validation training")
    parser.add_argument("--target-fold", type=str, default="all", help="Specific fold index to train (e.g. 0, 1, 2, 3, 4) or 'all' for all folds")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (only used with --cross-validate)")
    parser.add_argument("--eval-benchmark", action="store_true", help="Evaluate trained folds on the official held-out test set")
    parser.add_argument("--eval-cv-val", action="store_true", help="Evaluate in-fold validation splits with target-filtered metric")
    parser.add_argument("--resume", action="store_true", help="Resume training or hydration from previous checkpoints")
    parser.add_argument("--no-hf-sync", action="store_true", help="Skip syncing models to Hugging Face")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory containing datasets")
    parser.add_argument("--datasets", type=str, default="dentex,tufts", help="Comma-separated dataset names to combine (e.g. 'dentex', 'dentex,tufts')")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo for artifact sync")
    args = parser.parse_args()

    if args.eval_benchmark or args.eval_cv_val:
        evaluate_benchmark(args)
    elif args.cross_validate:
        cross_validate(args)
    else:
        dir_suffix = _dataset_dir_suffix(args.datasets)
        yaml_path = Path(f"data/yolo_{dir_suffix}/dataset.yaml").absolute()
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"Cannot find {yaml_path}. Run: python scripts/prepare_yolo_dataset.py --datasets {args.datasets}"
            )

        print(f"Loading YOLO model: {args.model}")
        train_single(str(yaml_path), args)


if __name__ == "__main__":
    main()
