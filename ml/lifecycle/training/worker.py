"""
Isolated Training Worker Process for Phase 14.2 Challenger Pipeline.

Executes XGBoost training and candidate bundle construction in a clean, dedicated subprocess.
Guarantees clean OpenMP runtime DLL state on Windows and complete process memory isolation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd

from ml.lifecycle.schemas import (
    ModelLifecycleStatus,
    ModelBundleManifest,
    calculate_file_sha256,
)
from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
    TARGET_COLUMN,
)
from ml.models.preprocessing import TreePreprocessor, prepare_features_and_target
from ml.models.xgboost_model import XGBoostFraudModel


def run_training_worker(job_config_path: Path) -> None:
    """
    Execute challenger training according to job specification.
    """
    start_time = time.perf_counter()
    with open(job_config_path, "r", encoding="utf-8") as f:
        job_config: Dict[str, Any] = json.load(f)

    dataset_path = Path(job_config["dataset_parquet_path"])
    output_dir = Path(job_config["output_dir"])
    model_version = str(job_config["model_version"]).strip()
    operating_threshold = float(job_config.get("operating_threshold", 0.78))
    hyperparameters = job_config.get("hyperparameters", {})
    training_metadata = job_config.get("training_metadata", {})

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    # Load required columns
    cols_to_load = PREDICTIVE_FEATURE_COLUMNS + [TARGET_COLUMN]
    df = pd.read_parquet(dataset_path, columns=cols_to_load)

    # 2. Extract X and y with strict schema validation
    X_df, y_series = prepare_features_and_target(df)
    del df

    # 3. Fit TreePreprocessor strictly on training data
    preprocessor = TreePreprocessor()
    preprocessor.fit(X_df)
    X_transformed = preprocessor.transform(X_df)

    # 4. Train XGBoost model
    xgb_model = XGBoostFraudModel(**hyperparameters)
    xgb_model.fit(X_transformed, y_series.to_numpy())

    # 5. Sanity check predictions
    sample_probs = xgb_model.predict_proba(X_transformed[:10])
    assert len(sample_probs) == 10, "Prediction sample length mismatch."
    assert np.all((sample_probs >= 0.0) & (sample_probs <= 1.0)), "Invalid probability range detected."

    fit_duration_sec = round(time.perf_counter() - start_time, 3)

    # 6. Save binary artifacts
    model_path = output_dir / "model.joblib"
    prep_path = output_dir / "preprocessor.joblib"
    manifest_path = output_dir / "manifest.json"
    checksums_path = output_dir / "checksums.json"

    joblib.dump(xgb_model, model_path, compress=3)
    joblib.dump(preprocessor, prep_path, compress=3)

    # 7. Compute cryptographic hashes
    model_sha = calculate_file_sha256(model_path)
    prep_sha = calculate_file_sha256(prep_path)

    checksums_dict = {
        "model.joblib": model_sha,
        "preprocessor.joblib": prep_sha,
    }
    with open(checksums_path, "w", encoding="utf-8") as f:
        json.dump(checksums_dict, f, indent=2)

    # 8. Construct ModelBundleManifest
    training_meta_payload = {
        **training_metadata,
        "fit_duration_seconds": fit_duration_sec,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "total_training_rows": len(y_series),
        "positive_fraud_rows": int((y_series == 1).sum()),
        "negative_legit_rows": int((y_series == 0).sum()),
        "feature_count": len(PREDICTIVE_FEATURE_COLUMNS),
    }

    manifest = ModelBundleManifest(
        model_version=model_version,
        model_family="xgboost",
        created_at=datetime.now(timezone.utc).isoformat(),
        status=ModelLifecycleStatus.CANDIDATE,
        operating_threshold=operating_threshold,
        hyperparameters=hyperparameters,
        training_metadata=training_meta_payload,
        policy_configuration={
            "policy_mode": "TRI_TIER",
            "review_threshold": 0.35,
            "block_threshold": operating_threshold,
        },
        sha256_checksums=checksums_dict,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json(indent=2))

    # 9. Write worker result
    result_path = output_dir / "worker_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "SUCCESS",
                "model_version": model_version,
                "model_path": str(model_path),
                "preprocessor_path": str(prep_path),
                "manifest_path": str(manifest_path),
                "checksums_path": str(checksums_path),
                "fit_duration_seconds": fit_duration_sec,
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Challenger Training Worker")
    parser.add_argument("--job-config", type=str, required=True, help="Path to job config JSON")
    args = parser.parse_args()

    config_path = Path(args.job_config)
    try:
        run_training_worker(config_path)
        sys.exit(0)
    except Exception as e:
        err_msg = traceback.format_exc()
        sys.stderr.write(f"Worker execution failed: {err_msg}\n")
        # Attempt to write error to result json if output_dir is readable
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            out_dir = Path(cfg["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "worker_result.json", "w", encoding="utf-8") as rf:
                json.dump({"status": "FAILED", "error": str(e), "traceback": err_msg}, rf, indent=2)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
