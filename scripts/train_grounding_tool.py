import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def train_single(yaml_path: str, args) -> dict:
    """Train a single YOLO model and return metrics."""
    resume = getattr(args, "resume", False)
    if resume:
        last_pt = Path("data/models/grounding_tool/weights/last.pt")
        if last_pt.exists():
            print(f"Resuming from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print("No checkpoint found — starting from scratch.")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="data/models",
        name="grounding_tool",
        exist_ok=True,
    )
    if hasattr(args, "patience") and args.patience:
        train_kwargs["patience"] = args.patience
    if resume:
        train_kwargs["resume"] = True

    print(f"Starting training on {args.device} for {args.epochs} epochs (patience={getattr(args, 'patience', 'N/A')})...")
    result = model.train(**train_kwargs)

    print("\nTraining Complete!")
    print(f"Model saved to: data/models/grounding_tool/weights/best.pt")

    return {}


def cross_validate(args):
    """Train YOLO independently on each CV fold, report metrics, select best."""
    cv_dir = Path("data/yolo_dentex_cv")
    summary_path = cv_dir / "fold_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot find {summary_path}. Run: python scripts/prepare_yolo_dataset.py --mode cv"
        )

    summary = json.loads(summary_path.read_text())
    n_folds = summary["n_folds"]
    results = []

    resume = getattr(args, "resume", False)

    for fold in range(n_folds):
        yaml_path = cv_dir / f"fold_{fold}" / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing {yaml_path}")

        fold_dir = Path("data/models") / f"cv_fold_{fold}"
        best_pt = fold_dir / "weights" / "best.pt"
        last_pt = fold_dir / "weights" / "last.pt"

        if resume and best_pt.exists():
            print(f"\n  FOLD {fold + 1}/{n_folds} — already complete, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        if resume and last_pt.exists():
            print(f"Resuming fold {fold} from {last_pt}")
            model = YOLO(str(last_pt))
        else:
            model = YOLO(args.model)

        train_kwargs = dict(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project="data/models",
            name=f"cv_fold_{fold}",
            exist_ok=True,
        )
        if hasattr(args, "patience") and args.patience:
            train_kwargs["patience"] = args.patience
        if resume and last_pt.exists():
            train_kwargs["resume"] = True

        result = model.train(**train_kwargs)

        metrics = {
            "fold": fold,
            "map50": float(result.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95": float(result.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(result.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(result.results_dict.get("metrics/recall(B)", 0)),
        }
        results.append(metrics)
        print(f"\nFold {fold}: mAP50={metrics['map50']:.4f}  mAP50-95={metrics['map50_95']:.4f}")

        # Free GPU memory before next fold
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- Find best fold by mAP50-95 ---
    best = max(results, key=lambda x: x["map50_95"])
    best_fold = best["fold"]

    # --- Copy best fold weights to final location ---
    best_src = Path(f"data/models/cv_fold_{best_fold}/weights/best.pt")
    best_dst = Path("data/models/grounding_tool_cv_best/weights/best.pt")
    best_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, best_dst)

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

    mean_map50 = sum(r["map50"] for r in results) / n_folds
    std_map50 = (sum((r["map50"] - mean_map50) ** 2 for r in results) / n_folds) ** 0.5
    mean_map50_95 = sum(r["map50_95"] for r in results) / n_folds
    std_map50_95 = (sum((r["map50_95"] - mean_map50_95) ** 2 for r in results) / n_folds) ** 0.5

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
    results_path = Path("data/models/grounding_tool_cv_best/cv_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"  Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Grounding Tool on DENTEX.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on (e.g., '0' for GPU 0, 'cpu' for CPU)")
    parser.add_argument("--cross-validate", action="store_true", help="Run 5-fold cross-validation instead of single training")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (only used with --cross-validate)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available")
    args = parser.parse_args()

    if args.cross_validate:
        cross_validate(args)
    else:
        yaml_path = Path("data/yolo_dentex/dataset.yaml").absolute()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Cannot find {yaml_path}. Run prepare_yolo_dataset.py first.")

        print(f"Loading YOLO model: {args.model}")
        train_single(str(yaml_path), args)


if __name__ == "__main__":
    main()
