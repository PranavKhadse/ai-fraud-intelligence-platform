"""
Phase 4 Machine Learning Pipeline: Full Training, Validation Benchmarking,
Threshold Analysis, Champion Selection, and Out-of-Time (OOT) Test Evaluation.

Enforces:
1. Strict process isolation on Windows between XGBoost, LightGBM, and Scikit-Learn.
2. Leakage-free training strictly on Train partition (1,296,675 rows).
3. Model comparison and threshold selection exclusively on Validation partition (277,859 rows).
4. Protected OOT Test partition (277,860 rows) evaluated strictly ONCE after freezing.
5. Primary metric: PR-AUC (Average Precision) due to severe class imbalance (~0.58% fraud).
"""

import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.models.config import (
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    TEST_FEATURES_PATH,
    ARTIFACTS_DIR,
    DOCS_DIR,
    TARGET_COLUMN,
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
    RANDOM_SEED,
    MODEL_CONFIGS,
)
from ml.models.runner import run_isolated_model_job
from ml.evaluation.metrics import compute_classification_metrics
from ml.evaluation.threshold_analysis import evaluate_threshold_sweep


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder supporting numpy types, timestamps, and paths."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def get_environment_info() -> Dict[str, Any]:
    """Capture environment, package versions, and system metadata for reproducibility."""
    import sklearn
    import xgboost
    import lightgbm
    import scipy
    
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "packages": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "random_seed": RANDOM_SEED,
    }



def train_and_evaluate_all_models(
    train_path: Path = TRAIN_FEATURES_PATH,
    val_path: Path = VAL_FEATURES_PATH,
    test_path: Path = TEST_FEATURES_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    figures_dir: Path = Path("docs/figures"),
) -> Dict[str, Any]:
    """
    Execute full Phase 4 training, validation benchmark, threshold selection, and OOT evaluation.
    """
    artifacts_dir = Path(artifacts_dir).resolve()
    figures_dir = Path(figures_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 4: FULL MODEL TRAINING & VALIDATION BENCHMARKING")
    print("=" * 80)
    
    # 1. Environment & Partition Inspection
    env_info = get_environment_info()
    print(f"Python: {env_info['python_version'].split()[0]} | Platform: {env_info['platform']}")
    print(f"Packages: {env_info['packages']}")
    
    # Load dataset partitions to inspect row counts & class distribution
    print("\nInspecting Dataset Partitions...")
    df_train = pd.read_parquet(train_path, columns=["timestamp", TARGET_COLUMN])
    df_val = pd.read_parquet(val_path, columns=["timestamp", TARGET_COLUMN])
    df_test = pd.read_parquet(test_path, columns=["timestamp", TARGET_COLUMN])
    
    train_rows, val_rows, test_rows = len(df_train), len(df_val), len(df_test)
    train_fraud = int(df_train[TARGET_COLUMN].sum())
    val_fraud = int(df_val[TARGET_COLUMN].sum())
    test_fraud = int(df_test[TARGET_COLUMN].sum())
    
    partition_stats = {
        "train": {
            "rows": train_rows,
            "fraud_count": train_fraud,
            "legit_count": train_rows - train_fraud,
            "fraud_rate": round(float(train_fraud / train_rows), 6),
            "start_time": str(df_train["timestamp"].min()),
            "end_time": str(df_train["timestamp"].max()),
        },
        "val": {
            "rows": val_rows,
            "fraud_count": val_fraud,
            "legit_count": val_rows - val_fraud,
            "fraud_rate": round(float(val_fraud / val_rows), 6),
            "start_time": str(df_val["timestamp"].min()),
            "end_time": str(df_val["timestamp"].max()),
        },
        "test": {
            "rows": test_rows,
            "fraud_count": test_fraud,
            "legit_count": test_rows - test_fraud,
            "fraud_rate": round(float(test_fraud / test_rows), 6),
            "start_time": str(df_test["timestamp"].min()),
            "end_time": str(df_test["timestamp"].max()),
        },
    }
    print(f"Train: {train_rows:,} rows (Fraud: {train_fraud:,}, {partition_stats['train']['fraud_rate']:.4%})")
    print(f"Val:   {val_rows:,} rows (Fraud: {val_fraud:,}, {partition_stats['val']['fraud_rate']:.4%})")
    print(f"Test:  {test_rows:,} rows (Fraud: {test_fraud:,}, {partition_stats['test']['fraud_rate']:.4%})")
    print(f"Predictive Feature Count: {len(PREDICTIVE_FEATURE_COLUMNS)}")
    
    # 2. Train All 4 Models in Process Isolation
    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    model_results: Dict[str, Dict[str, Any]] = {}
    validation_benchmarks: Dict[str, Dict[str, Any]] = {}
    threshold_sweeps: Dict[str, Dict[str, Any]] = {}
    
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING THE FOUR MODELS (PROCESS ISOLATED)")
    print("=" * 80)
    
    for m_name in model_names:
        print(f"\n>>> Training {m_name.upper()}...")
        # Train and validate strictly without passing test partition
        res = run_isolated_model_job(
            model_name=m_name,
            train_features_path=train_path,
            val_features_path=val_path,
            test_features_path=None,
            save_artifacts_dir=artifacts_dir,
        )
        model_results[m_name] = res
        fit_time = res["fit_time_sec"]
        print(f"    [Done] Fit time: {fit_time:.2f}s")
        
        # Validation Evaluation (Default Threshold 0.5)
        y_val = res["y_val"]
        val_prob = res["val_prob"]
        val_metrics = compute_classification_metrics(y_val, val_prob, threshold=0.5)
        
        # Validation Threshold Sweep (0.01 - 0.99)
        sweep = evaluate_threshold_sweep(y_val, val_prob, step=0.01)
        
        val_metrics["fit_time_sec"] = round(fit_time, 2)
        val_metrics["best_f1_threshold"] = sweep["best_f1_threshold"]
        val_metrics["best_f1_value"] = round(sweep["best_f1_value"], 5)
        
        validation_benchmarks[m_name] = val_metrics
        threshold_sweeps[m_name] = sweep
        
        print(f"    Validation PR-AUC:  {val_metrics['pr_auc']:.5f} (PRIMARY)")
        print(f"    Validation ROC-AUC: {val_metrics['roc_auc']:.5f}")
        print(f"    Validation F1 (0.5): {val_metrics['f1']:.5f} | Precision: {val_metrics['precision']:.5f} | Recall: {val_metrics['recall']:.5f}")
        print(f"    Best F1: {sweep['best_f1_value']:.5f} at Threshold {sweep['best_f1_threshold']:.2f}")

    # 3. Validation Model Comparison Table
    print("\n" + "=" * 80)
    print("STEP 2: VALIDATION BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<22} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Prec(0.5)':<10} | {'Rec(0.5)':<9} | {'F1(0.5)':<8} | {'Best F1':<8} | {'Best Tau':<8} | {'Fit (s)':<8}")
    print("-" * 105)
    for m_name in model_names:
        bm = validation_benchmarks[m_name]
        print(f"{m_name:<22} | {bm['pr_auc']:<8.5f} | {bm['roc_auc']:<8.5f} | {bm['precision']:<10.5f} | {bm['recall']:<9.5f} | {bm['f1']:<8.5f} | {bm['best_f1_value']:<8.5f} | {bm['best_f1_threshold']:<8.2f} | {bm['fit_time_sec']:<8.1f}")

    # 4. Model Selection & Threshold Freezing
    print("\n" + "=" * 80)
    print("STEP 3: CHAMPION SELECTION & THRESHOLD FREEZING (VALIDATION ONLY)")
    print("=" * 80)
    
    # Primary selection metric: PR-AUC
    champion_name = max(model_names, key=lambda m: validation_benchmarks[m]["pr_auc"])
    champion_val_pr_auc = validation_benchmarks[champion_name]["pr_auc"]
    champion_sweep = threshold_sweeps[champion_name]
    frozen_threshold = champion_sweep["best_f1_threshold"]
    frozen_metrics = champion_sweep["best_f1_metrics"]
    
    print(f"Selected Champion Model: {champion_name.upper()}")
    print(f"Champion Validation PR-AUC: {champion_val_pr_auc:.5f}")
    print(f"Champion Selected Threshold (tau*): {frozen_threshold:.2f}")
    print(f"Validation Performance at tau*={frozen_threshold:.2f}:")
    print(f"  - Precision: {frozen_metrics['precision']:.5f}")

    print(f"  - Recall:    {frozen_metrics['recall']:.5f}")
    print(f"  - F1-Score:  {frozen_metrics['f1']:.5f}")
    print(f"  - TP: {frozen_metrics['confusion_matrix']['tp']:,} | FP: {frozen_metrics['confusion_matrix']['fp']:,} | FN: {frozen_metrics['confusion_matrix']['fn']:,} | TN: {frozen_metrics['confusion_matrix']['tn']:,}")
    
    # Copy champion artifacts
    shutil.copy(artifacts_dir / f"{champion_name}_model.joblib", artifacts_dir / "champion_model.joblib")
    shutil.copy(artifacts_dir / f"{champion_name}_preprocessor.joblib", artifacts_dir / "champion_preprocessor.joblib")
    print(f"Persisted champion model to {artifacts_dir / 'champion_model.joblib'}")


    # 5. Final Out-Of-Time (OOT) Test Evaluation
    print("\n" + "=" * 80)
    print("STEP 4: FINAL OUT-OF-TIME (OOT) TEST EVALUATION")
    print("=" * 80)
    print(f"Evaluating frozen champion ({champion_name.upper()}) at frozen threshold tau*={frozen_threshold:.2f} on UNSEEN Test holdout...")
    
    # Run champion evaluation on test holdout
    test_run_res = run_isolated_model_job(
        model_name=champion_name,
        train_features_path=train_path,
        val_features_path=val_path,
        test_features_path=test_path,
        save_artifacts_dir=None,
    )
    y_test = test_run_res["y_test"]
    test_prob = test_run_res["test_prob"]
    
    # Compute test metrics at frozen threshold and default 0.5
    test_metrics_frozen = compute_classification_metrics(y_test, test_prob, threshold=frozen_threshold)
    test_metrics_default = compute_classification_metrics(y_test, test_prob, threshold=0.5)
    
    print("\nFINAL OOT TEST EVALUATION RESULTS:")
    print(f"  - Test PR-AUC (Average Precision): {test_metrics_frozen['pr_auc']:.5f}")
    print(f"  - Test ROC-AUC:                    {test_metrics_frozen['roc_auc']:.5f}")
    print(f"  - At Frozen Threshold (tau* = {frozen_threshold:.2f}):")
    print(f"      * Precision: {test_metrics_frozen['precision']:.5f}")
    print(f"      * Recall:    {test_metrics_frozen['recall']:.5f}")
    print(f"      * F1-Score:  {test_metrics_frozen['f1']:.5f}")
    print(f"      * Confusion Matrix: TP={test_metrics_frozen['confusion_matrix']['tp']:,}, FP={test_metrics_frozen['confusion_matrix']['fp']:,}, FN={test_metrics_frozen['confusion_matrix']['fn']:,}, TN={test_metrics_frozen['confusion_matrix']['tn']:,}")
    print(f"      * Predicted Fraud Count: {test_metrics_frozen['support']['predicted_fraud']:,} / {test_rows:,}")
    print(f"      * Actual Fraud Count:    {test_metrics_frozen['support']['fraud']:,} / {test_rows:,}")
    print(f"  - At Default Threshold (tau = 0.50):")
    print(f"      * Precision: {test_metrics_default['precision']:.5f}")
    print(f"      * Recall:    {test_metrics_default['recall']:.5f}")
    print(f"      * F1-Score:  {test_metrics_default['f1']:.5f}")

    # 6. Feature Importance Summary
    print("\n" + "=" * 80)
    print("STEP 5: FEATURE IMPORTANCE EXTRACTION")
    print("=" * 80)
    feature_importances = {
        m: model_results[m]["feature_importances"]
        for m in model_names
    }
    
    champion_fi = feature_importances[champion_name]
    print(f"Top 15 Features for Champion Model ({champion_name.upper()}):")
    for rank, item in enumerate(champion_fi[:15], 1):
        print(f"  {rank:2d}. {item['feature']:<35} : {item['importance']:.6f}")

    # Plot Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top20_fi = champion_fi[:20][::-1]
    y_pos = np.arange(len(top20_fi))
    ax.barh(y_pos, [x["importance"] for x in top20_fi], color="#1f77b4", edgecolor="#0d47a1")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([x["feature"] for x in top20_fi], fontsize=9)
    ax.set_xlabel("Relative Feature Importance (Gain / Metric)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top 20 Predictive Features - Champion Model ({champion_name.upper()})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fi_plot_path = figures_dir / "feature_importance.png"
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to {fi_plot_path}")

    # Plot Threshold Sweep Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    sweep_df = pd.DataFrame(champion_sweep["sweep_table"])
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#2ca02c", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#1f77b4", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1-Score", color="#d62728", linewidth=2.5)
    ax.axvline(frozen_threshold, color="#7f7f7f", linestyle="--", label=f"Selected tau* = {frozen_threshold:.2f}")
    ax.scatter([frozen_threshold], [frozen_metrics["f1"]], color="#d62728", s=100, zorder=5)
    ax.set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_title(f"Validation Threshold Optimization Curve - {champion_name.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    thresh_plot_path = figures_dir / "threshold_analysis.png"
    plt.savefig(thresh_plot_path, dpi=300)
    plt.close()
    print(f"Saved threshold analysis chart to {thresh_plot_path}")

    # 7. Save Comprehensive JSON Metadata Artifacts
    model_metadata = {
        "model_version": "1.0.0",
        "champion_model": champion_name,
        "champion_hyperparameters": MODEL_CONFIGS[champion_name],
        "random_seed": RANDOM_SEED,
        "selected_threshold": frozen_threshold,
        "environment": env_info,
        "partition_statistics": partition_stats,
        "feature_count": len(PREDICTIVE_FEATURE_COLUMNS),
        "feature_names": PREDICTIVE_FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_PREDICTORS,
        "validation_benchmark": validation_benchmarks[champion_name],
        "oot_test_metrics_frozen_threshold": test_metrics_frozen,
        "oot_test_metrics_default_threshold": test_metrics_default,
        "training_timestamp": str(pd.Timestamp.now()),
        "saved_artifacts": [
            "champion_model.joblib",
            "champion_preprocessor.joblib",
            "model_metadata.json",
            "all_models_validation_benchmark.json",
            "threshold_analysis.json",
            "feature_importance.json",
        ],
    }
    
    with open(artifacts_dir / "model_metadata.json", "w") as f:
        json.dump(model_metadata, f, indent=2, cls=NumpyEncoder)
        
    with open(artifacts_dir / "all_models_validation_benchmark.json", "w") as f:
        json.dump(validation_benchmarks, f, indent=2, cls=NumpyEncoder)
        
    # Clean threshold sweep results for JSON serialization
    clean_sweeps = {}
    for m, sw in threshold_sweeps.items():
        clean_sweeps[m] = {
            "best_f1_threshold": sw["best_f1_threshold"],
            "best_f1_value": sw["best_f1_value"],
            "best_f1_metrics": sw["best_f1_metrics"],
            "operating_points": sw["operating_points"],
            "sweep_table": sw["sweep_table"],
        }
    with open(artifacts_dir / "threshold_analysis.json", "w") as f:
        json.dump(clean_sweeps, f, indent=2, cls=NumpyEncoder)
        
    with open(artifacts_dir / "feature_importance.json", "w") as f:
        json.dump(feature_importances, f, indent=2, cls=NumpyEncoder)

        
    print(f"\nAll Phase 4 model artifacts successfully saved to {artifacts_dir}")
    
    return {
        "env_info": env_info,
        "partition_stats": partition_stats,
        "validation_benchmarks": validation_benchmarks,
        "threshold_sweeps": threshold_sweeps,
        "champion_name": champion_name,
        "frozen_threshold": frozen_threshold,
        "test_metrics_frozen": test_metrics_frozen,
        "test_metrics_default": test_metrics_default,
        "feature_importances": feature_importances,
    }


if __name__ == "__main__":
    train_and_evaluate_all_models()
