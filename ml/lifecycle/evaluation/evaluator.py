"""
Candidate Model Evaluator for Phase 14 Model Lifecycle & Controlled Promotion.

Provides decoupled, objective evaluation of candidate models against frozen
validation partitions:
1. Validates bundle manifest, semantic version, and physical SHA-256 artifact integrity.
2. Validates 55-feature predictive schema and XGBoost algorithm family.
3. Computes PR-AUC, ROC-AUC, Precision, Recall, F1, FPR, BLOCK Precision, and Expected Cost.
4. Evaluates candidate metrics against objective quality gates.
5. Emits deterministic, strongly typed CandidateEvaluationResult.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backend.app.repositories.model_registry_repository import ModelRegistryRepository

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.evaluation.gates import CandidateGateConfig, default_candidate_gate_config, evaluate_candidate_gates
from ml.lifecycle.evaluation.metrics import calculate_evaluation_metrics, calculate_latency_distribution
from ml.lifecycle.evaluation.schemas import CandidateEvaluationResult, EvaluationScope
from ml.lifecycle.schemas import (
    ModelBundleManifest,
    ModelLifecycleStatus,
    validate_bundle_manifest,
    validate_semantic_version,
    verify_bundle_integrity,
)
from ml.models.config import VAL_FEATURES_PATH
from ml.models.preprocessing import prepare_features_and_target


class CandidateModelEvaluator:
    """
    Decoupled lifecycle evaluation engine for candidate model bundles.
    """

    def __init__(
        self,
        lifecycle_config: Optional[LifecycleConfig] = None,
        gate_config: Optional[CandidateGateConfig] = None,
        cost_config: Optional[CostConfig] = None,
    ) -> None:
        """
        Initialize the candidate model evaluator with lifecycle and gate configurations.
        """
        self.lifecycle_config = lifecycle_config or default_lifecycle_config
        self.gate_config = gate_config or default_candidate_gate_config
        self.cost_config = cost_config or CostConfig()

    def load_and_validate_bundle(
        self, bundle_dir: Union[str, Path]
    ) -> Tuple[Any, Any, ModelBundleManifest]:
        """
        Load a candidate model bundle and verify manifest, semver, checksums, and feature schema.

        Args:
            bundle_dir: Filesystem directory containing manifest.json, model.joblib, and preprocessor.joblib.

        Returns:
            Tuple[model, preprocessor, manifest]

        Raises:
            FileNotFoundError: If the bundle or required artifacts do not exist.
            ValueError: If manifest schema, checksums, model family, or feature count is invalid.
        """
        b_path = Path(bundle_dir).resolve()
        manifest_file = b_path / "manifest.json"

        if not b_path.exists() or not b_path.is_dir():
            raise FileNotFoundError(f"Model bundle directory not found: {b_path}")
        if not manifest_file.exists():
            raise FileNotFoundError(f"Bundle manifest.json missing in directory: {b_path}")

        # 1. Load and validate manifest
        manifest = ModelBundleManifest.load(manifest_file)
        validate_bundle_manifest(manifest)

        # 2. Validate Semantic Versioning
        if not validate_semantic_version(manifest.model_version):
            raise ValueError(
                f"Candidate bundle version '{manifest.model_version}' violates Semantic Versioning MAJOR.MINOR.PATCH."
            )

        # 3. Validate Supported Model Family
        if manifest.model_family not in self.lifecycle_config.supported_model_families:
            raise ValueError(
                f"Unsupported model family '{manifest.model_family}'. "
                f"Supported families: {self.lifecycle_config.supported_model_families}"
            )

        # 4. Verify Physical Checksums and Artifact Integrity
        is_valid, integrity_errors = verify_bundle_integrity(b_path, manifest)
        if not is_valid:
            error_msg = "; ".join(integrity_errors)
            raise ValueError(f"Candidate bundle integrity verification failed: {error_msg}")

        # 5. Load model and preprocessor artifacts
        model_path = b_path / "model.joblib"
        preproc_path = b_path / "preprocessor.joblib"

        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)

        # 6. Validate preprocessor feature schema
        if hasattr(preprocessor, "feature_names"):
            feat_count = len(preprocessor.feature_names)
            if feat_count != self.lifecycle_config.expected_feature_count:
                raise ValueError(
                    f"Candidate preprocessor feature count ({feat_count}) does not match "
                    f"platform expected feature count ({self.lifecycle_config.expected_feature_count})."
                )

        return model, preprocessor, manifest

    def evaluate_candidate(
        self,
        bundle_dir: Union[str, Path],
        dataset_path: Optional[Union[str, Path]] = None,
        threshold_override: Optional[float] = None,
        scope: EvaluationScope = EvaluationScope.VALIDATION,
        evaluated_by: str = "candidate_evaluator",
        benchmark_latency: bool = True,
    ) -> CandidateEvaluationResult:
        """
        Evaluate a candidate model bundle against the frozen validation partition.

        Args:
            bundle_dir: Path to candidate bundle directory.
            dataset_path: Optional dataset path (strictly defaults to VAL_FEATURES_PATH).
            threshold_override: Optional operating decision threshold override in (0, 1).
            scope: Evaluation dataset scope (default: VALIDATION).
            evaluated_by: Governance actor running evaluation.
            benchmark_latency: If True, measures inference latency percentiles.

        Returns:
            CandidateEvaluationResult containing metrics, gate checks, and pass/fail status.
        """
        # 1. Load and validate bundle integrity
        model, preprocessor, manifest = self.load_and_validate_bundle(bundle_dir)

        # 2. Resolve dataset path (strictly validation partition)
        data_file = Path(dataset_path).resolve() if dataset_path else Path(VAL_FEATURES_PATH).resolve()
        if not data_file.exists():
            raise FileNotFoundError(f"Evaluation dataset partition not found at: {data_file}")

        # 3. Load dataset and extract 55 features + target
        df = pd.read_parquet(data_file)
        X, y = prepare_features_and_target(df)

        if len(X.columns) != self.lifecycle_config.expected_feature_count:
            raise ValueError(
                f"Prepared feature matrix has {len(X.columns)} features, expected "
                f"{self.lifecycle_config.expected_feature_count}."
            )

        # 4. Transform features using candidate preprocessor
        X_trans = preprocessor.transform(X)

        # 5. Measure latency distribution if requested
        latency_stats: Optional[Dict[str, float]] = None
        if benchmark_latency and len(X_trans) > 0:
            latencies_ms: List[float] = []
            # Benchmark 50 individual inference calls across sample
            sample_indices = np.linspace(0, len(X_trans) - 1, num=min(50, len(X_trans)), dtype=int)
            for idx in sample_indices:
                single_row = X_trans[idx : idx + 1]
                t0 = time.perf_counter()
                _ = model.predict_proba(single_row)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
            latency_stats = calculate_latency_distribution(latencies_ms)

        # 6. Generate full dataset probabilities
        raw_prob = model.predict_proba(X_trans)
        if hasattr(raw_prob, "ndim") and raw_prob.ndim == 2 and raw_prob.shape[1] > 1:
            y_prob = raw_prob[:, 1].astype(np.float64)
        else:
            y_prob = np.ravel(raw_prob).astype(np.float64)

        # 7. Determine operating threshold
        if threshold_override is not None:
            operating_threshold = float(threshold_override)
        elif manifest.operating_threshold is not None:
            operating_threshold = float(manifest.operating_threshold)
        else:
            operating_threshold = float(self.lifecycle_config.default_operating_threshold)

        # 8. Calculate comprehensive evaluation metrics
        metrics = calculate_evaluation_metrics(
            y_true=y,
            y_prob=y_prob,
            threshold=operating_threshold,
            cost_config=self.cost_config,
        )

        # 9. Evaluate candidate against quality gates
        overall_passed, gate_results = evaluate_candidate_gates(
            metrics=metrics,
            gate_config=self.gate_config,
            latency_stats=latency_stats,
        )

        # 10. Construct strongly typed CandidateEvaluationResult
        return CandidateEvaluationResult(
            model_version=manifest.model_version,
            model_family=manifest.model_family,
            scope=scope,
            operating_threshold=operating_threshold,
            metrics=metrics,
            gate_results=gate_results,
            overall_passed=overall_passed,
            evaluated_by=evaluated_by,
            bundle_directory=str(bundle_dir),
            dataset_path=str(data_file),
            sample_count=len(y),
            fraud_count=int(np.sum(y == 1)),
            latency_stats=latency_stats,
        )

    async def evaluate_registered_candidate(
        self,
        repo: ModelRegistryRepository,
        model_version: str,
        dataset_path: Optional[Union[str, Path]] = None,
        update_registry_metrics: bool = False,
        threshold_override: Optional[float] = None,
        evaluated_by: str = "candidate_evaluator",
    ) -> CandidateEvaluationResult:
        """
        Evaluate a model version registered in the database repository.

        Args:
            repo: Active ModelRegistryRepository instance.
            model_version: Target semantic model version to evaluate.
            dataset_path: Optional dataset path (defaults to VAL_FEATURES_PATH).
            update_registry_metrics: If True, persists validation metrics into the registry entry.
            threshold_override: Optional threshold override.
            evaluated_by: User or system author running evaluation.

        Returns:
            CandidateEvaluationResult
        """
        entry = await repo.get_by_version(model_version)
        if entry is None:
            raise ValueError(f"Model version '{model_version}' is not registered in the database.")

        if entry.bundle_directory is None:
            raise ValueError(
                f"Model version '{model_version}' has no bundle_directory registered in the database."
            )

        bundle_path = Path(entry.bundle_directory)
        result = self.evaluate_candidate(
            bundle_dir=bundle_path,
            dataset_path=dataset_path,
            threshold_override=threshold_override or float(entry.operating_threshold),
            evaluated_by=evaluated_by,
        )

        # Optionally record validation metrics back into database record (without promoting)
        if update_registry_metrics:
            entry.validation_metrics = result.metrics.model_dump()
            await repo._session.flush()

        return result
