import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
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
    dataset_list = [d.strip() for d in datasets_arg.split(",") if d.strip()]
    return "_".join(dataset_list) if dataset_list != ["dentex"] else "dentex"


def _model_subdir_prefix(dir_suffix: str) -> str:
    """Empty for the default "dentex"-only case (preserves every existing
    dentex-only run's exact output paths -- cv_fold_0, grounding_tool_cv_best,
    etc. -- so nothing already on disk or already backed up to HF silently
    stops being found). Non-default combos (e.g. "dentex_tufts") get their
    own prefixed subtree instead of overwriting/conflating with the
    dentex-only one, the same collision risk dataset_tag exists to prevent
    in prepare_yolo_dataset.py's convert_single_image."""
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

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(model_root),
        name=f"{prefix}grounding_tool",
        exist_ok=True,
    )
    if hasattr(args, "patience") and args.patience:
        train_kwargs["patience"] = args.patience
    if resume:
        train_kwargs["resume"] = True

    print(f"Starting training on {args.device} for {args.epochs} epochs (patience={getattr(args, 'patience', 'N/A')})...")
    result = model.train(**train_kwargs)

    print("\nTraining Complete!")
    print(f"Model saved to: {model_root / 'grounding_tool' / 'weights' / 'best.pt'}")

    return {}


def cross_validate(args):
    """Train YOLO independently on each CV fold, report metrics, select best."""
    model_root = get_model_root()
    dir_suffix = _dataset_dir_suffix(getattr(args, "datasets", "dentex"))
    prefix = _model_subdir_prefix(dir_suffix)
    cv_dir = Path(f"data/yolo_{dir_suffix}_cv")
    summary_path = cv_dir / "fold_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot find {summary_path}. Run: python scripts/prepare_yolo_dataset.py "
            f"--mode cv --datasets {getattr(args, 'datasets', 'dentex')}"
        )

    summary = json.loads(summary_path.read_text())
    n_folds = summary["n_folds"]
    results = []

    resume = getattr(args, "resume", False)

    for fold in range(n_folds):
        yaml_path = cv_dir / f"fold_{fold}" / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing {yaml_path}")

        fold_dir = model_root / f"{prefix}cv_fold_{fold}"
        best_pt = fold_dir / "weights" / "best.pt"
        last_pt = fold_dir / "weights" / "last.pt"

        # A fold is complete only if YOLO finished and generated the final validation images
        confusion_matrix = fold_dir / "confusion_matrix.png"
        if resume and confusion_matrix.exists():
            print(f"\n  FOLD {fold + 1}/{n_folds} — already complete, skipping.")
            
            # Since we skipped training, we must read the metrics from results.csv so it's included in the final mean!
            csv_path = fold_dir / "results.csv"
            if csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                last_row = df.iloc[-1]
                metrics = {
                    "fold": fold,
                    "map50": float(last_row["metrics/mAP50(B)"]),
                    "map50_95": float(last_row["metrics/mAP50-95(B)"]),
                    "precision": float(last_row["metrics/precision(B)"]),
                    "recall": float(last_row["metrics/recall(B)"]),
                }
                results.append(metrics)
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
            project=str(model_root),
            name=f"{prefix}cv_fold_{fold}",
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
    best_src = model_root / f"{prefix}cv_fold_{best_fold}" / "weights" / "best.pt"
    best_dst = model_root / f"{prefix}grounding_tool_cv_best" / "weights" / "best.pt"
    best_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, best_dst)
    
    # Copy all training results (PR curves, confusion matrix, args.yaml, results.csv, etc.)
    best_fold_dir = model_root / f"{prefix}cv_fold_{best_fold}"
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
    results_path = model_root / f"{prefix}grounding_tool_cv_best" / "cv_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"  Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Grounding Tool.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--device", type=str, default="0", help="Device to run on (e.g., '0' for GPU 0, 'cpu' for CPU)")
    parser.add_argument("--cross-validate", action="store_true", help="Run 5-fold cross-validation instead of single training")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (only used with --cross-validate)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available")
    parser.add_argument(
        "--datasets", type=str, default="dentex",
        help="Comma-separated dataset names -- must match whatever --datasets value was "
             "passed to prepare_yolo_dataset.py when building the input directory this reads "
             "(default: dentex, unchanged behavior/paths). e.g. --datasets dentex,tufts. "
             "Also tags this run's own output directories (cv_fold_N, grounding_tool_cv_best, "
             "etc.) so a combined run's results/checkpoints never overwrite or get mistaken "
             "for a dentex-only run's -- see _model_subdir_prefix's docstring.",
    )
    args = parser.parse_args()

    if args.cross_validate:
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
