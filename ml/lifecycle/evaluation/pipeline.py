"""
Full-Pipeline Candidate Evaluation & Threshold Analysis Orchestrator for Phase 14.3.

Coordinates the complete candidate evaluation lifecycle:
1. Validation inference & latency distribution profiling.
2. Deterministic validation threshold sweep and selection of optimal operating threshold tau*.
3. Freezing tau* for all subsequent evaluation.
4. Operational decision routing analysis via existing RiskEvaluator on validation set.
5. Protected out-of-time (OOT) benchmark evaluation strictly at frozen tau*.
6. Candidate manifest update (preserving training provenance).
7. Optional PostgreSQL database registry synchronization.
"""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backend.app.repositories.model_registry_repository import ModelRegistryRepository

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.evaluation.metrics import calculate_evaluation_metrics, calculate_latency_distribution
from ml.lifecycle.evaluation.operational import OperationalDecisionEvaluator
from ml.lifecycle.evaluation.schemas import (
    FullCandidateEvaluationResult,
    OperationalDecisionMetrics,
    ThresholdOptimizationObjective,
    ThresholdSelectionResult,
)
from ml.lifecycle.evaluation.threshold import CandidateThresholdAnalyzer
from ml.lifecycle.schemas import (
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
    validate_bundle_manifest,
    validate_semantic_version,
    verify_bundle_integrity,
)
from ml.models.config import TEST_FEATURES_PATH, VAL_FEATURES_PATH
from ml.models.preprocessing import prepare_features_and_target
from ml.risk_engine.config import PolicyMode
from ml.risk_engine.rules import RuleEngine


class FullCandidateEvaluationPipeline:
    """
    Orchestrates full-pipeline candidate model evaluation, validation threshold optimization,
    and protected OOT benchmarking.
    """

    def __init__(
        self,
        lifecycle_config: Optional[LifecycleConfig] = None,
        cost_config: Optional[CostConfig] = None,
        threshold_analyzer: Optional[CandidateThresholdAnalyzer] = None,
        operational_evaluator: Optional[OperationalDecisionEvaluator] = None,
    ) -> None:
        """
        Initialize the full candidate evaluation pipeline.
        """
        self.lifecycle_config = lifecycle_config or default_lifecycle_config
        self.cost_config = cost_config or CostConfig()
        self.threshold_analyzer = threshold_analyzer or CandidateThresholdAnalyzer(cost_config=self.cost_config)
        self.operational_evaluator = operational_evaluator or OperationalDecisionEvaluator()

    def load_and_validate_candidate_bundle(
        self, bundle_dir: Union[str, Path]
    ) -> Tuple[Any, Any, ModelBundleManifest]:
        """
        Load candidate bundle and verify physical SHA-256 integrity, SemVer, model family, and 55 features.
        """
        b_path = Path(bundle_dir).resolve()
        manifest_file = b_path / "manifest.json"

        if not b_path.exists() or not b_path.is_dir():
            raise FileNotFoundError(f"Candidate bundle directory not found: {b_path}")
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest missing at: {manifest_file}")

        manifest = ModelBundleManifest.load(manifest_file)
        validate_bundle_manifest(manifest)

        if not validate_semantic_version(manifest.model_version):
            raise ValueError(f"Invalid model version '{manifest.model_version}'. Must follow semantic versioning.")
        if manifest.model_family not in self.lifecycle_config.supported_model_families:
            raise ValueError(f"Unsupported model family '{manifest.model_family}'.")

        is_valid, errors = verify_bundle_integrity(b_path, manifest)
        if not is_valid:
            raise ValueError(f"Bundle integrity verification failed: {'; '.join(errors)}")

        model_path = b_path / "model.joblib"
        preproc_path = b_path / "preprocessor.joblib"

        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)

        if hasattr(preprocessor, "feature_names"):
            feat_count = len(preprocessor.feature_names)
            if feat_count != self.lifecycle_config.expected_feature_count:
                raise ValueError(
                    f"Candidate preprocessor has {feat_count} features, expected {self.lifecycle_config.expected_feature_count}."
                )

        return model, preprocessor, manifest

    def evaluate_candidate_bundle(
        self,
        bundle_dir: Union[str, Path],
        val_dataset_path: Optional[Union[str, Path]] = None,
        oot_dataset_path: Optional[Union[str, Path]] = None,
        objective: ThresholdOptimizationObjective = ThresholdOptimizationObjective.MIN_EXPECTED_COST,
        evaluate_operational: bool = True,
        policy_mode: PolicyMode = PolicyMode.TRI_TIER,
        rule_engine: Optional[RuleEngine] = None,
        evaluated_by: str = "full_candidate_evaluator",
        update_manifest: bool = True,
        benchmark_latency: bool = True,
    ) -> FullCandidateEvaluationResult:
        """
        Execute full-pipeline evaluation for a candidate model bundle.

        Sequence:
        1. Load & validate candidate bundle integrity.
        2. Validation inference & latency measurement.
        3. Validation threshold sweep -> select optimal tau* -> freeze tau*.
        4. Operational decision evaluation on validation data via RiskEvaluator.
        5. Protected OOT evaluation strictly at frozen tau*.
        6. Operational decision evaluation on OOT data at frozen tau*.
        7. Update candidate bundle manifest.json (preserving training provenance).
        """
        b_path = Path(bundle_dir).resolve()
        model, preprocessor, manifest = self.load_and_validate_candidate_bundle(b_path)

        # 1. Resolve and load validation dataset
        val_file = Path(val_dataset_path).resolve() if val_dataset_path else Path(VAL_FEATURES_PATH).resolve()
        if not val_file.exists():
            raise FileNotFoundError(f"Validation dataset partition not found at: {val_file}")

        val_df = pd.read_parquet(val_file)
        X_val, y_val = prepare_features_and_target(val_df)

        if len(X_val.columns) != self.lifecycle_config.expected_feature_count:
            raise ValueError(
                f"Validation feature matrix has {len(X_val.columns)} columns, expected {self.lifecycle_config.expected_feature_count}."
            )

        # 2. Transform validation features and measure latency
        X_val_trans = preprocessor.transform(X_val)

        latency_stats: Optional[Dict[str, float]] = None
        if benchmark_latency and len(X_val_trans) > 0:
            latencies_ms: List[float] = []
            sample_indices = np.linspace(0, len(X_val_trans) - 1, num=min(50, len(X_val_trans)), dtype=int)
            for idx in sample_indices:
                single_row = X_val_trans[idx : idx + 1]
                t0 = time.perf_counter()
                _ = model.predict_proba(single_row)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
            latency_stats = calculate_latency_distribution(latencies_ms)

        # 3. Generate validation probabilities
        raw_prob_val = model.predict_proba(X_val_trans)
        if hasattr(raw_prob_val, "ndim") and raw_prob_val.ndim == 2 and raw_prob_val.shape[1] > 1:
            y_prob_val = raw_prob_val[:, 1].astype(np.float64)
        else:
            y_prob_val = np.ravel(raw_prob_val).astype(np.float64)

        # 4. Perform validation threshold sweep and select tau* (ZERO OOT ACCESS)
        threshold_selection = self.threshold_analyzer.evaluate_sweep(
            y_true=y_val,
            y_prob=y_prob_val,
            objective=objective,
        )
        tau_star = threshold_selection.selected_threshold

        # 5. Compute comprehensive validation performance metrics at frozen tau*
        validation_metrics = calculate_evaluation_metrics(
            y_true=y_val,
            y_prob=y_prob_val,
            threshold=tau_star,
            cost_config=self.cost_config,
        )

        # 6. Optional validation operational routing via existing RiskEvaluator
        val_operational_metrics: Optional[OperationalDecisionMetrics] = None
        if evaluate_operational:
            val_operational_metrics = self.operational_evaluator.evaluate_operational_decisions(
                df=val_df,
                model_path=b_path / "model.joblib",
                preprocessor_path=b_path / "preprocessor.joblib",
                operating_threshold=tau_star,
                metadata_path=b_path / "manifest.json",
                policy_mode=policy_mode,
                rule_engine=rule_engine,
            )

        # 7. Protected Out-Of-Time (OOT) evaluation strictly using frozen tau*
        oot_file = Path(oot_dataset_path).resolve() if oot_dataset_path else Path(TEST_FEATURES_PATH).resolve()
        if not oot_file.exists():
            raise FileNotFoundError(f"Protected OOT dataset partition not found at: {oot_file}")

        oot_df = pd.read_parquet(oot_file)
        X_oot, y_oot = prepare_features_and_target(oot_df)

        if len(X_oot.columns) != self.lifecycle_config.expected_feature_count:
            raise ValueError(
                f"OOT feature matrix has {len(X_oot.columns)} columns, expected {self.lifecycle_config.expected_feature_count}."
            )

        X_oot_trans = preprocessor.transform(X_oot)
        raw_prob_oot = model.predict_proba(X_oot_trans)
        if hasattr(raw_prob_oot, "ndim") and raw_prob_oot.ndim == 2 and raw_prob_oot.shape[1] > 1:
            y_prob_oot = raw_prob_oot[:, 1].astype(np.float64)
        else:
            y_prob_oot = np.ravel(raw_prob_oot).astype(np.float64)

        # Compute OOT metrics strictly at frozen tau*
        oot_metrics = calculate_evaluation_metrics(
            y_true=y_oot,
            y_prob=y_prob_oot,
            threshold=tau_star,
            cost_config=self.cost_config,
        )

        # Optional OOT operational routing
        oot_operational_metrics: Optional[OperationalDecisionMetrics] = None
        if evaluate_operational:
            oot_operational_metrics = self.operational_evaluator.evaluate_operational_decisions(
                df=oot_df,
                model_path=b_path / "model.joblib",
                preprocessor_path=b_path / "preprocessor.joblib",
                operating_threshold=tau_star,
                metadata_path=b_path / "manifest.json",
                policy_mode=policy_mode,
                rule_engine=rule_engine,
            )

        # 8. Update manifest.json while strictly preserving training provenance
        if update_manifest:
            manifest.operating_threshold = tau_star
            manifest.validation_metrics = validation_metrics.model_dump()
            manifest.oot_holdout_metrics = oot_metrics.model_dump()
            manifest.evaluation_configuration = {
                "optimization_objective": objective.value,
                "step": self.threshold_analyzer.step,
                "cost_config": {
                    "fp_cost": self.cost_config.false_positive_cost,
                    "fn_cost": self.cost_config.false_negative_cost,
                    "tp_cost": self.cost_config.true_positive_cost,
                    "tn_cost": self.cost_config.true_negative_cost,
                },
                "evaluated_by": evaluated_by,
            }
            manifest.policy_configuration = {
                "policy_mode": policy_mode.value,
                "operating_threshold": tau_star,
            }
            manifest.save(b_path / "manifest.json")

        return FullCandidateEvaluationResult(
            model_version=manifest.model_version,
            model_family=manifest.model_family,
            selected_operating_threshold=tau_star,
            threshold_selection=threshold_selection,
            validation_metrics=validation_metrics,
            validation_operational_metrics=val_operational_metrics,
            oot_metrics=oot_metrics,
            oot_operational_metrics=oot_operational_metrics,
            latency_stats=latency_stats,
            evaluated_by=evaluated_by,
            bundle_directory=str(b_path),
            validation_dataset_path=str(val_file),
            oot_dataset_path=str(oot_file),
            validation_sample_count=len(y_val),
            validation_fraud_count=int(np.sum(y_val == 1)),
            oot_sample_count=len(y_oot),
            oot_fraud_count=int(np.sum(y_oot == 1)),
        )

    async def evaluate_registered_candidate(
        self,
        repo: ModelRegistryRepository,
        model_version: str,
        val_dataset_path: Optional[Union[str, Path]] = None,
        oot_dataset_path: Optional[Union[str, Path]] = None,
        objective: ThresholdOptimizationObjective = ThresholdOptimizationObjective.MIN_EXPECTED_COST,
        evaluate_operational: bool = True,
        policy_mode: PolicyMode = PolicyMode.TRI_TIER,
        rule_engine: Optional[RuleEngine] = None,
        update_db: bool = True,
        evaluated_by: str = "full_candidate_evaluator",
    ) -> FullCandidateEvaluationResult:
        """
        Evaluate a registered candidate model version and optionally synchronize results to the database.

        Guarantees:
        - Candidate status remains ModelLifecycleStatus.CANDIDATE.
        - is_active_champion remains False.
        - Updates operating_threshold, validation_metrics, oot_metrics, and policy_configuration.
        """
        entry = await repo.get_by_version(model_version)
        if entry is None:
            raise ValueError(f"Model version '{model_version}' is not registered in the database.")
        if entry.bundle_directory is None:
            raise ValueError(f"Model version '{model_version}' has no bundle_directory registered in the database.")

        result = self.evaluate_candidate_bundle(
            bundle_dir=Path(entry.bundle_directory),
            val_dataset_path=val_dataset_path,
            oot_dataset_path=oot_dataset_path,
            objective=objective,
            evaluate_operational=evaluate_operational,
            policy_mode=policy_mode,
            rule_engine=rule_engine,
            evaluated_by=evaluated_by,
            update_manifest=True,
        )

        if update_db:
            entry.operating_threshold = Decimal(str(result.selected_operating_threshold))
            entry.validation_metrics = result.validation_metrics.model_dump()
            entry.oot_metrics = result.oot_metrics.model_dump()
            entry.policy_configuration = {
                "policy_mode": policy_mode.value,
                "operating_threshold": result.selected_operating_threshold,
            }
            # Crucial invariant: Ensure status remains CANDIDATE and is_active_champion is False
            entry.is_active_champion = False
            entry.status = ModelLifecycleStatus.CANDIDATE.value

            await repo._session.flush()

        return result
