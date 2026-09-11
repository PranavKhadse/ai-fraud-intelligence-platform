"""
Process-Isolated Model Training and Inference Runner for Phase 4 ML Pipeline.

Guarantees:
- Safe execution of tree-based models on Windows by preventing cross-library OpenMP TLS memory collisions.
- XGBoost and LightGBM execute in clean, dedicated worker sub-processes.
- Unified interface for training, probability prediction, and feature importance extraction.
"""

import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import joblib


def run_isolated_model_job(
    model_name: str,
    train_features_path: Path,
    val_features_path: Path,
    test_features_path: Optional[Path] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    save_artifacts_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Launch model training and evaluation inside a clean, isolated worker sub-process.
    Prevents cross-DLL OpenMP runtime pollution between XGBoost and LightGBM on Windows.
    """
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
        tmp_output_path = Path(tmp_file.name)
        
    cmd = [
        sys.executable,
        "-m",
        "ml.models.worker",
        "--model",
        model_name,
        "--train",
        str(train_features_path),
        "--val",
        str(val_features_path),
        "--output",
        str(tmp_output_path),
    ]
    
    if test_features_path is not None:
        cmd.extend(["--test", str(test_features_path)])
        
    if hyperparams is not None:
        cmd.extend(["--hyperparams", json.dumps(hyperparams)])
        
    if save_artifacts_dir is not None:
        cmd.extend(["--save-artifacts-dir", str(save_artifacts_dir)])
        
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if res.returncode != 0:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise RuntimeError(
            f"Isolated worker for model '{model_name}' failed with code {res.returncode}:\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
        
    result = joblib.load(tmp_output_path)
    if tmp_output_path.exists():
        tmp_output_path.unlink()
        
    return result

