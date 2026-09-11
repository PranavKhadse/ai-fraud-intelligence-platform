"""
Isolated Model Worker Process Entrypoint for Phase 4 ML Pipeline.

Executes a single model training and inference job in a clean, dedicated process.
Ensures zero cross-library OpenMP DLL runtime collisions between XGBoost and LightGBM on Windows.
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from ml.models.config import (
    MODEL_CONFIGS,
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
    TARGET_COLUMN,
)



def run_worker(
    model_name: str,
    train_path: Path,
    val_path: Path,
    test_path: Optional[Path] = None,
    hyperparams_json: Optional[str] = None,
    output_path: Optional[Path] = None,
    save_artifacts_dir: Optional[Path] = None,
) -> None:
    # 1. Load dataset partitions directly with required columns only
    cols_to_load = PREDICTIVE_FEATURE_COLUMNS + [TARGET_COLUMN]
    df_train = pd.read_parquet(train_path, columns=cols_to_load)
    from ml.models.preprocessing import prepare_features_and_target
    X_train_df, y_train = prepare_features_and_target(df_train)
    del df_train
    
    df_val = pd.read_parquet(val_path, columns=cols_to_load)
    X_val_df, y_val = prepare_features_and_target(df_val)
    del df_val
    
    if test_path and Path(test_path).exists():
        df_test = pd.read_parquet(test_path, columns=cols_to_load)
        X_test_df, y_test = prepare_features_and_target(df_test)
        del df_test
    else:
        X_test_df, y_test = None, None


    
    cfg = MODEL_CONFIGS[model_name].copy()
    if hyperparams_json:
        overrides = json.loads(hyperparams_json) if isinstance(hyperparams_json, str) else hyperparams_json
        cfg.update(overrides)
        
    start_time = pd.Timestamp.now()
    
    if model_name == "logistic_regression":
        from ml.models.preprocessing import LinearPreprocessor
        from ml.models.baseline import BaselineLogisticRegression
        preproc = LinearPreprocessor().fit(X_train_df)
        X_train_mat = preproc.transform(X_train_df)
        X_val_mat = preproc.transform(X_val_df)
        X_test_mat = preproc.transform(X_test_df) if X_test_df is not None else None
        
        model = BaselineLogisticRegression(**cfg).fit(X_train_mat, y_train)
        fit_time_sec = (pd.Timestamp.now() - start_time).total_seconds()
        
        val_prob = model.predict_proba(X_val_mat)
        test_prob = model.predict_proba(X_test_mat) if X_test_mat is not None else None
        fi = model.get_feature_importance(PREDICTIVE_FEATURE_COLUMNS)
        
    elif model_name == "random_forest":
        from ml.models.preprocessing import TreePreprocessor
        from ml.models.random_forest import RandomForestBaseline
        preproc = TreePreprocessor().fit(X_train_df)
        X_train_mat = preproc.transform(X_train_df)
        X_val_mat = preproc.transform(X_val_df)
        X_test_mat = preproc.transform(X_test_df) if X_test_df is not None else None
        
        model = RandomForestBaseline(**cfg).fit(X_train_mat, y_train)
        fit_time_sec = (pd.Timestamp.now() - start_time).total_seconds()
        
        val_prob = model.predict_proba(X_val_mat)
        test_prob = model.predict_proba(X_test_mat) if X_test_mat is not None else None
        fi = model.get_feature_importance(PREDICTIVE_FEATURE_COLUMNS)
        
    elif model_name == "xgboost":
        from ml.models.preprocessing import TreePreprocessor
        from ml.models.xgboost_model import XGBoostFraudModel
        preproc = TreePreprocessor().fit(X_train_df)
        X_train_mat = preproc.transform(X_train_df)
        X_val_mat = preproc.transform(X_val_df)
        X_test_mat = preproc.transform(X_test_df) if X_test_df is not None else None
        
        model = XGBoostFraudModel(**cfg).fit(X_train_mat, y_train)
        fit_time_sec = (pd.Timestamp.now() - start_time).total_seconds()
        
        val_prob = model.predict_proba(X_val_mat)
        test_prob = model.predict_proba(X_test_mat) if X_test_mat is not None else None
        fi = model.get_feature_importance(PREDICTIVE_FEATURE_COLUMNS)
        
    elif model_name == "lightgbm":
        # Pure pandas categorical encoding avoiding sklearn OpenMP DLL load
        X_train_enc = X_train_df.copy()
        X_val_enc = X_val_df.copy()
        X_test_enc = X_test_df.copy() if X_test_df is not None else None
        
        category_mappings = {}
        for cat_col in CATEGORICAL_PREDICTORS:
            cats = sorted(X_train_df[cat_col].dropna().unique())
            mapping = {val: float(i) for i, val in enumerate(cats)}
            category_mappings[cat_col] = mapping
            X_train_enc[cat_col] = X_train_df[cat_col].map(mapping).fillna(-1.0).astype(np.float64)
            X_val_enc[cat_col] = X_val_df[cat_col].map(mapping).fillna(-1.0).astype(np.float64)
            if X_test_enc is not None:
                X_test_enc[cat_col] = X_test_df[cat_col].map(mapping).fillna(-1.0).astype(np.float64)
                
        X_train_mat = np.ascontiguousarray(X_train_enc.values, dtype=np.float64)
        X_val_mat = np.ascontiguousarray(X_val_enc.values, dtype=np.float64)
        X_test_mat = np.ascontiguousarray(X_test_enc.values, dtype=np.float64) if X_test_enc is not None else None
        
        from ml.models.lightgbm_model import LightGBMFraudModel
        model = LightGBMFraudModel(**cfg).fit(X_train_mat, y_train)
        fit_time_sec = (pd.Timestamp.now() - start_time).total_seconds()
        
        val_prob = model.predict_proba(X_val_mat)
        test_prob = model.predict_proba(X_test_mat) if X_test_mat is not None else None
        fi = model.get_feature_importance(PREDICTIVE_FEATURE_COLUMNS)
        preproc = category_mappings
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    if save_artifacts_dir:
        save_dir = Path(save_artifacts_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir / f"{model_name}_model.joblib")
        joblib.dump(preproc, save_dir / f"{model_name}_preprocessor.joblib")
        
    result = {
        "model_name": model_name,
        "fit_time_sec": fit_time_sec,
        "val_prob": val_prob,
        "y_val": y_val.values if hasattr(y_val, "values") else y_val,
        "test_prob": test_prob,
        "y_test": y_test.values if hasattr(y_test, "values") and y_test is not None else None,
        "feature_importances": fi,
        "config": cfg,
    }
    
    if output_path:
        joblib.dump(result, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated Model Worker")
    parser.add_argument("--model", required=True, type=str, help="Model name")
    parser.add_argument("--train", required=True, type=str, help="Path to train features parquet")
    parser.add_argument("--val", required=True, type=str, help="Path to val features parquet")
    parser.add_argument("--test", required=False, default=None, type=str, help="Path to test features parquet")
    parser.add_argument("--hyperparams", required=False, default=None, type=str, help="JSON hyperparams override")
    parser.add_argument("--output", required=True, type=str, help="Path to output joblib artifact")
    parser.add_argument("--save-artifacts-dir", required=False, default=None, type=str, help="Directory to save model & preprocessor artifacts")
    
    args = parser.parse_args()
    run_worker(
        model_name=args.model,
        train_path=Path(args.train),
        val_path=Path(args.val),
        test_path=Path(args.test) if args.test else None,
        hyperparams_json=args.hyperparams,
        output_path=Path(args.output),
        save_artifacts_dir=Path(args.save_artifacts_dir) if args.save_artifacts_dir else None,
    )

