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
    if "," in datasets_arg:
        dataset_list = [d.strip() for d in datasets_arg.split(",") if d.strip()]
    else:
        dataset_list = [d.strip() for d in datasets_arg.split() if d.strip()]
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
        if resume and confusion_matrix.exists():
            print(f"\n  FOLD {fold + 1}/{n_folds} — already complete, skipping.")
            
            csv_path = fold_dir / "results.csv"
            if csv_path.exists():
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
            continue

        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        if resume and last_pt.exists():
            print(f"Resuming fold {fold} from {last_pt}")
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
            device=args.device,
            project=str(model_root),
            name=f"{prefix}cv_fold_{fold}",
            exist_ok=True,
        )
        if hasattr(args, "save_period") and args.save_period:
            train_kwargs["save_period"] = args.save_period
        if hasattr(args, "patience") and args.patience:
            train_kwargs["patience"] = args.patience
        if resume and last_pt.exists():
            train_kwargs["resume"] = True

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
    # 1. Download missing fold checkpoints from HF Hub if available
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo = os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models")
    if hf_token and not hf_token.startswith("YOUR_"):
        try:
            from huggingface_hub import snapshot_download
            print(f"Checking Hugging Face Hub ({hf_repo}/yolo_cv) for any remote fold checkpoints...")
            staging_dir = model_root / "_hf_staging"
            snapshot_download(
                repo_id=hf_repo,
                repo_type="model",
                allow_patterns=["yolo_cv/*"],
                local_dir=str(staging_dir),
                token=hf_token,
            )
            yolo_cv_staged = staging_dir / "yolo_cv"
            if yolo_cv_staged.exists():
                for item in yolo_cv_staged.glob("*"):
                    dest = model_root / item.name
                    if not dest.exists():
                        shutil.copytree(item, dest)
                    else:
                        for f in item.rglob("*"):
                            if f.is_file():
                                rel = f.relative_to(item)
                                dst_f = dest / rel
                                if not dst_f.exists():
                                    dst_f.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(f, dst_f)
            shutil.rmtree(staging_dir, ignore_errors=True)
        except Exception as e:
            print(f"HF auto-hydration notice: {e}")

    # 2. Check for any folds in alternate directory locations
    for fold in range(n_folds):
        target_fold_dir = model_root / f"{prefix}cv_fold_{fold}"
        if not target_fold_dir.exists():
            candidates = [
                Path("data/yolo_cv") / f"{prefix}cv_fold_{fold}",
                Path("data/yolo_cv") / f"cv_fold_{fold}",
                model_root / "yolo_cv" / f"{prefix}cv_fold_{fold}",
                model_root / f"cv_fold_{fold}",
            ]
            for cand in candidates:
                if cand.exists():
                    shutil.copytree(cand, target_fold_dir)
                    break

    found_folds = []
    for fold in range(n_folds):
        weight_path = model_root / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt"
        if weight_path.exists():
            found_folds.append(fold)
    return found_folds


def evaluate_benchmark(args):
    """Evaluate all trained fold models across both 5-fold CV splits and the held-out test set."""
    model_root = get_model_root()
    dir_suffix = _dataset_dir_suffix(getattr(args, "datasets", "dentex,tufts"))
    prefix = _model_subdir_prefix(dir_suffix)
    cv_dir = Path(f"data/yolo_{dir_suffix}_cv")
    test_yaml = cv_dir / "test" / "dataset.yaml"

    if not test_yaml.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_yaml}. Run prepare_yolo_dataset.py --mode cv first.")

    summary_path = cv_dir / "fold_summary.json"
    n_folds = 5
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        n_folds = summary.get("n_folds", 5)

    found_folds = _ensure_folds_hydrated(model_root, prefix, n_folds)
    print(f"Located {len(found_folds)}/{n_folds} trained fold models in {model_root}")

    # --- Part 1: 5-Fold Cross-Validation Metrics Table ---
    cv_results = []
    for fold in range(n_folds):
        fold_dir = model_root / f"{prefix}cv_fold_{fold}"
        fold_yaml = cv_dir / f"fold_{fold}" / "dataset.yaml"
        metrics = {"fold": fold, "map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0}

        csv_path = fold_dir / "results.csv"
        if csv_path.exists():
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

        best_pt = fold_dir / "weights" / "best.pt"
        if best_pt.exists() and (metrics["map50"] == 0.0 and metrics["map50_95"] == 0.0) and fold_yaml.exists():
            try:
                model = YOLO(str(best_pt))
                val_res = model.val(data=str(fold_yaml), split="val", batch=args.batch, imgsz=args.imgsz, device=args.device, verbose=False)
                res_d = getattr(val_res, "results_dict", {})
                metrics["map50"] = float(res_d.get("metrics/mAP50(B)", 0.0))
                metrics["map50_95"] = float(res_d.get("metrics/mAP50-95(B)", 0.0))
                metrics["precision"] = float(res_d.get("metrics/precision(B)", 0.0))
                metrics["recall"] = float(res_d.get("metrics/recall(B)", 0.0))
            except Exception as e:
                print(f"Validation notice for fold {fold}: {e}")

        if best_pt.exists() or metrics["map50"] > 0:
            cv_results.append(metrics)

    if cv_results:
        print(f"\n{'=' * 60}")
        print(f"  {len(cv_results)}-FOLD CROSS-VALIDATION RESULTS (Internal Validation)")
        print(f"{'=' * 60}")
        print(f"  {'Fold':<6} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
        print(f"  {'-' * 46}")
        for r in cv_results:
            print(
                f"  {r['fold']:<6} {r['map50']:<10.4f} {r['map50_95']:<10.4f} "
                f"{r['precision']:<10.4f} {r['recall']:<10.4f}"
            )
        print(f"  {'-' * 46}")
        mean_map50 = sum(r["map50"] for r in cv_results) / len(cv_results)
        std_map50 = (sum((r["map50"] - mean_map50) ** 2 for r in cv_results) / len(cv_results)) ** 0.5
        mean_map50_95 = sum(r["map50_95"] for r in cv_results) / len(cv_results)
        std_map50_95 = (sum((r["map50_95"] - mean_map50_95) ** 2 for r in cv_results) / len(cv_results)) ** 0.5
        print(f"  Mean mAP50:      {mean_map50:.4f} +/- {std_map50:.4f}")
        print(f"  Mean mAP50-95:   {mean_map50_95:.4f} +/- {std_map50_95:.4f}")

        cv_best = max(cv_results, key=lambda x: x["map50_95"])
        cv_data = {
            "folds": cv_results,
            "best_fold": cv_best["fold"],
            "mean_map50": mean_map50,
            "std_map50": std_map50,
            "mean_map50_95": mean_map50_95,
            "std_map50_95": std_map50_95,
        }
        cv_out = model_root / f"{prefix}grounding_tool_cv_best" / "cv_results.json"
        cv_out.parent.mkdir(parents=True, exist_ok=True)
        cv_out.write_text(json.dumps(cv_data, indent=2))
        print(f"  Saved CV results to: {cv_out}")

    # --- Part 2: Official Held-Out Test Set Evaluation ---
    benchmark_results = []
    print(f"\n{'=' * 65}")
    print("  HELD-OUT TEST SET EVALUATION (Official DENTEX Benchmark)")
    print(f"{'=' * 65}")
    print(f"  {'Model / Fold':<25} {'mAP50':<10} {'mAP50-95':<10} {'Precision':<10} {'Recall':<10}")
    print(f"  {'-' * 65}")

    for fold in range(n_folds):
        for weight_type in ("best", "last"):
            weight_path = model_root / f"{prefix}cv_fold_{fold}" / "weights" / f"{weight_type}.pt"
            if not weight_path.exists():
                continue
            model = YOLO(str(weight_path))
            val_res = model.val(data=str(test_yaml), split="val", batch=args.batch, imgsz=args.imgsz, device=args.device, verbose=False)
            res_dict = getattr(val_res, "results_dict", {})
            metrics = {
                "name": f"Fold {fold} ({weight_type})",
                "fold": fold,
                "type": weight_type,
                "map50": float(res_dict.get("metrics/mAP50(B)", 0)),
                "map50_95": float(res_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(res_dict.get("metrics/precision(B)", 0)),
                "recall": float(res_dict.get("metrics/recall(B)", 0)),
            }
            benchmark_results.append(metrics)
            print(
                f"  {metrics['name']:<25} {metrics['map50']:<10.4f} {metrics['map50_95']:<10.4f} "
                f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}"
            )

    # Save benchmark results & copy top performer weights to grounding_tool_cv_best
    if benchmark_results:
        best_benchmark = max(benchmark_results, key=lambda x: x["map50_95"])
        print(f"  {'-' * 65}")
        print(f"  Top Benchmark Performer: {best_benchmark['name']} (mAP50: {best_benchmark['map50']:.4f}, mAP50-95: {best_benchmark['map50_95']:.4f})")

        winning_fold = best_benchmark["fold"]
        winning_type = best_benchmark["type"]
        win_src = model_root / f"{prefix}cv_fold_{winning_fold}" / "weights" / f"{winning_type}.pt"
        best_dst = model_root / f"{prefix}grounding_tool_cv_best" / "weights" / "best.pt"
        best_dst.parent.mkdir(parents=True, exist_ok=True)
        if win_src.exists():
            shutil.copy2(win_src, best_dst)
            print(f"  Assigned top benchmark model ({best_benchmark['name']}) -> {best_dst}")

        # Keep every fold's best model in fold_best_models/
        for fold in range(n_folds):
            src = model_root / f"{prefix}cv_fold_{fold}" / "weights" / "best.pt"
            if src.exists():
                ensemble_dst = model_root / f"{prefix}fold_best_models" / f"fold_{fold}_best.pt"
                ensemble_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, ensemble_dst)

        out_json = model_root / f"{prefix}grounding_tool_cv_best" / "benchmark_evaluation.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(benchmark_results, indent=2))
        print(f"  Benchmark results saved to: {out_json}")

        # Sync best models and benchmark results to Hugging Face
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
                        commit_message="Final YOLO 5-Fold CV results & benchmark weights",
                    )
                    print(f"  [HF Sync] Successfully uploaded all fold checkpoints and results to {hf_repo}/yolo_cv")
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
    parser.add_argument("--eval-benchmark", action="store_true", help="Evaluate trained folds on the official held-out test set")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (only used with --cross-validate)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available")
    parser.add_argument("--no-hf-sync", action="store_true", help="Disable automatic syncing to Hugging Face Hub")
    parser.add_argument(
        "--datasets", type=str, default="dentex,tufts",
        help="Comma-separated dataset names (e.g. dentex,tufts).",
    )
    args = parser.parse_args()

    if args.eval_benchmark:
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
