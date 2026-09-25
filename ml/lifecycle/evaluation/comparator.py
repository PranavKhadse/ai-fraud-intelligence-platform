"""
Champion vs. Challenger Comparator Engine for Phase 14.4.

Executes side-by-side, measurement-only comparative evaluations between
active production Champion v1.0.0 and Candidate v1.1.0 across identical
validation and protected out-of-time (OOT) partitions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from ml.cost_optimization.config import CostConfig
from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.evaluation.comparison_schemas import (
    ChampionChallengerComparisonResult,
    ClassificationComparison,
    DeltaSign,
    LatencyComparison,
    MetricDelta,
    OperationalComparison,
    compute_metric_delta,
)
from ml.lifecycle.evaluation.metrics import (
    calculate_evaluation_metrics,
    calculate_latency_distribution,
)
from ml.lifecycle.evaluation.operational import OperationalDecisionEvaluator
from ml.lifecycle.evaluation.pipeline import (
    prepare_features_and_target,
)
from ml.lifecycle.evaluation.schemas import OperationalDecisionMetrics
from ml.lifecycle.schemas import (
    EvaluationMetricsSummary,
    ModelBundleManifest,
    ModelLifecycleStatus,
    calculate_file_sha256,
    verify_bundle_integrity,
)
from ml.models.config import TEST_FEATURES_PATH, VAL_FEATURES_PATH
from ml.risk_engine.config import PolicyMode
from ml.risk_engine.rules import RuleEngine

logger = logging.getLogger("ChampionChallengerComparator")


class ChampionImmutabilityViolationError(Exception):
    """Raised if active Champion model artifacts are modified during comparison."""
    pass


class ChampionMetadataDiscrepancyError(Exception):
    """Raised if conflicting threshold values are detected in Champion metadata sources."""
    pass


class ChampionChallengerComparator:
    """
    Objective, measurement-only model comparison engine.
    
    Evaluates Champion and Candidate independently and computes exact mathematical
    deltas across classification performance, expected financial cost, operational
    routing distributions, and standardized inference latencies.
    """

    def __init__(
        self,
        lifecycle_config: Optional[LifecycleConfig] = None,
        cost_config: Optional[CostConfig] = None,
    ) -> None:
        self.config = lifecycle_config or default_lifecycle_config
        self.cost_config = cost_config or CostConfig()
        self.operational_evaluator = OperationalDecisionEvaluator()

    def _snapshot_champion_checksums(
        self,
        model_path: Path,
        preprocessor_path: Path,
        metadata_path: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Compute cryptographic SHA-256 hashes of Champion artifacts to detect tampering."""
        checksums: Dict[str, str] = {}
        if not model_path.exists() or not model_path.is_file():
            raise FileNotFoundError(f"Champion model artifact not found at: {model_path}")
        if not preprocessor_path.exists() or not preprocessor_path.is_file():
            raise FileNotFoundError(f"Champion preprocessor artifact not found at: {preprocessor_path}")

        checksums["model"] = calculate_file_sha256(model_path)
        checksums["preprocessor"] = calculate_file_sha256(preprocessor_path)

        if metadata_path and metadata_path.exists() and metadata_path.is_file():
            checksums["metadata"] = calculate_file_sha256(metadata_path)

        return checksums

    def _verify_champion_immutability(
        self,
        initial_checksums: Dict[str, str],
        model_path: Path,
        preprocessor_path: Path,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Assert that Champion artifacts remain 100% byte-identical."""
        current_checksums = self._snapshot_champion_checksums(
            model_path, preprocessor_path, metadata_path
        )
        for name, initial_sha in initial_checksums.items():
            current_sha = current_checksums.get(name)
            if initial_sha != current_sha:
                raise ChampionImmutabilityViolationError(
                    f"Champion immutability violation detected! Artifact '{name}' SHA-256 changed "
                    f"from {initial_sha} to {current_sha} during comparative evaluation."
                )

    def load_authoritative_champion(
        self,
        champion_source: Optional[Union[str, Path, Dict[str, Path]]] = None,
    ) -> Tuple[Any, Any, Dict[str, Any], float, Dict[str, Path]]:
        """
        Dynamically load Champion model, preprocessor, metadata, and extract its
        authoritative operating threshold without hard-coding any values.

        Returns:
            Tuple of (model, preprocessor, metadata_dict, operating_threshold, paths_dict).
        """
        # Resolve artifact paths
        if isinstance(champion_source, dict):
            m_path = Path(champion_source["model"]).resolve()
            p_path = Path(champion_source["preprocessor"]).resolve()
            meta_path = Path(champion_source.get("metadata", self.config.champion_metadata_path)).resolve()
        elif isinstance(champion_source, (str, Path)):
            base_dir = Path(champion_source).resolve()
            if base_dir.is_file():
                base_dir = base_dir.parent
            m_path = base_dir / "champion_model.joblib"
            if not m_path.exists():
                m_path = base_dir / "model.joblib"
            p_path = base_dir / "champion_preprocessor.joblib"
            if not p_path.exists():
                p_path = base_dir / "preprocessor.joblib"
            meta_path = base_dir / "model_metadata.json"
            if not meta_path.exists():
                meta_path = base_dir / "manifest.json"
        else:
            m_path = self.config.champion_model_path.resolve()
            p_path = self.config.champion_preprocessor_path.resolve()
            meta_path = self.config.champion_metadata_path.resolve()

        if not m_path.exists():
            raise FileNotFoundError(f"Champion model artifact not found at: {m_path}")
        if not p_path.exists():
            raise FileNotFoundError(f"Champion preprocessor artifact not found at: {p_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Champion metadata artifact not found at: {meta_path}")

        # Load artifacts
        model = joblib.load(m_path)
        preprocessor = joblib.load(p_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Extract and verify authoritative threshold across all available keys
        threshold_candidates: List[Tuple[str, float]] = []

        if "selected_threshold" in metadata and metadata["selected_threshold"] is not None:
            threshold_candidates.append(("selected_threshold", float(metadata["selected_threshold"])))
        if "operating_threshold" in metadata and metadata["operating_threshold"] is not None:
            threshold_candidates.append(("operating_threshold", float(metadata["operating_threshold"])))
        if "validation_benchmark" in metadata and isinstance(metadata["validation_benchmark"], dict):
            val_b = metadata["validation_benchmark"]
            if "best_f1_threshold" in val_b and val_b["best_f1_threshold"] is not None:
                threshold_candidates.append(("validation_benchmark.best_f1_threshold", float(val_b["best_f1_threshold"])))
        if "oot_test_metrics_frozen_threshold" in metadata and isinstance(metadata["oot_test_metrics_frozen_threshold"], dict):
            oot_b = metadata["oot_test_metrics_frozen_threshold"]
            if "threshold" in oot_b and oot_b["threshold"] is not None:
                threshold_candidates.append(("oot_test_metrics_frozen_threshold.threshold", float(oot_b["threshold"])))

        # Also inspect sibling threshold_analysis.json if present in same directory
        analysis_path = meta_path.parent / "threshold_analysis.json"
        if analysis_path.exists():
            try:
                with open(analysis_path, "r", encoding="utf-8") as f_a:
                    a_data = json.load(f_a)
                if "xgboost" in a_data and isinstance(a_data["xgboost"], dict):
                    xg_entry = a_data["xgboost"]
                    if "best_f1_threshold" in xg_entry and xg_entry["best_f1_threshold"] is not None:
                        threshold_candidates.append(
                            ("threshold_analysis.json/xgboost/best_f1_threshold", float(xg_entry["best_f1_threshold"]))
                        )
            except Exception as e:
                logger.warning(f"Could not parse sibling threshold_analysis.json: {e}")

        if not threshold_candidates:
            raise ValueError(
                f"No operating threshold found in Champion metadata at {meta_path}. Cannot establish authoritative threshold."
            )

        # Verify cross-source consistency
        primary_key, primary_threshold = threshold_candidates[0]
        for key, val in threshold_candidates[1:]:
            if abs(primary_threshold - val) > 1e-4:
                raise ChampionMetadataDiscrepancyError(
                    f"Discrepancy detected in Champion metadata threshold sources! "
                    f"'{primary_key}'={primary_threshold} vs '{key}'={val}."
                )

        if not (0.0 < primary_threshold < 1.0):
            raise ValueError(f"Extracted Champion threshold {primary_threshold} is out of valid bounds (0.0, 1.0).")

        paths_dict = {
            "model": m_path,
            "preprocessor": p_path,
            "metadata": meta_path,
        }

        return model, preprocessor, metadata, primary_threshold, paths_dict

    def load_candidate_bundle(
        self,
        bundle_dir: Union[str, Path],
    ) -> Tuple[Any, Any, ModelBundleManifest, float, Dict[str, Path]]:
        """
        Load Candidate model, preprocessor, manifest, and extract its frozen operating threshold.
        """
        b_path = Path(bundle_dir).resolve()
        if not b_path.exists() or not b_path.is_dir():
            raise FileNotFoundError(f"Candidate bundle directory not found: {b_path}")

        m_path = b_path / "model.joblib"
        p_path = b_path / "preprocessor.joblib"
        manifest_path = b_path / "manifest.json"

        if not m_path.exists() or not p_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Candidate bundle missing core artifacts. Expected model.joblib, preprocessor.joblib, manifest.json in {b_path}"
            )

        manifest = ModelBundleManifest.load(manifest_path)
        model = joblib.load(m_path)
        preprocessor = joblib.load(p_path)

        frozen_threshold = float(manifest.operating_threshold)
        if not (0.0 < frozen_threshold < 1.0):
            raise ValueError(
                f"Invalid candidate frozen operating threshold: {frozen_threshold}. Must be in (0.0, 1.0)."
            )

        paths_dict = {
            "model": m_path,
            "preprocessor": p_path,
            "manifest": manifest_path,
        }

        return model, preprocessor, manifest, frozen_threshold, paths_dict

    def benchmark_identical_latency(
        self,
        champ_model: Any,
        champ_prep: Any,
        cand_model: Any,
        cand_prep: Any,
        X_val: pd.DataFrame,
        sample_count: int = 50,
        warmup_count: int = 5,
    ) -> LatencyComparison:
        """
        Execute identical latency benchmark across Champion and Candidate:
        - Same deterministic slice of N rows
        - Same 5 unmeasured warmup iterations per model
        - Single-threaded single-row timing with time.perf_counter()
        - Standard percentile calculations
        """
        num_samples = min(sample_count, len(X_val))
        sample_indices = np.linspace(0, len(X_val) - 1, num=num_samples, dtype=int)
        X_slice = X_val.iloc[sample_indices]

        # 1. Transform slices
        X_champ_trans = champ_prep.transform(X_slice)
        X_cand_trans = cand_prep.transform(X_slice)

        # 2. Champion Warm-up & Measurement
        for _ in range(warmup_count):
            _ = champ_model.predict_proba(X_champ_trans[0:1])

        champ_latencies_ms: List[float] = []
        for i in range(num_samples):
            single_row = X_champ_trans[i : i + 1]
            t0 = time.perf_counter()
            _ = champ_model.predict_proba(single_row)
            t1 = time.perf_counter()
            champ_latencies_ms.append((t1 - t0) * 1000.0)

        # 3. Candidate Warm-up & Measurement
        for _ in range(warmup_count):
            _ = cand_model.predict_proba(X_cand_trans[0:1])

        cand_latencies_ms: List[float] = []
        for i in range(num_samples):
            single_row = X_cand_trans[i : i + 1]
            t0 = time.perf_counter()
            _ = cand_model.predict_proba(single_row)
            t1 = time.perf_counter()
            cand_latencies_ms.append((t1 - t0) * 1000.0)

        champ_stats = calculate_latency_distribution(champ_latencies_ms)
        cand_stats = calculate_latency_distribution(cand_latencies_ms)

        deltas: Dict[str, MetricDelta] = {}
        for key in ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]:
            deltas[key] = compute_metric_delta(
                metric_name=key,
                champion_val=champ_stats[key],
                candidate_val=cand_stats[key],
                is_rate=False,
            )

        return LatencyComparison(
            champion_latency=champ_stats,
            candidate_latency=cand_stats,
            deltas=deltas,
            warmup_iterations=warmup_count,
            measured_samples=num_samples,
        )

    def compare_classification_metrics(
        self,
        champ_metrics: EvaluationMetricsSummary,
        cand_metrics: EvaluationMetricsSummary,
    ) -> ClassificationComparison:
        """
        Compute neutral mathematical deltas between Champion and Candidate classification metrics.
        """
        rate_metrics = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fpr", "accuracy"]
        count_metrics = ["expected_cost", "tp", "fp", "tn", "fn", "threshold", "total_samples"]

        deltas: Dict[str, MetricDelta] = {}

        for m in rate_metrics:
            c_val = getattr(champ_metrics, m, None)
            cand_val = getattr(cand_metrics, m, None)
            if c_val is not None and cand_val is not None:
                deltas[m] = compute_metric_delta(
                    metric_name=m,
                    champion_val=c_val,
                    candidate_val=cand_val,
                    is_rate=True,
                )

        for m in count_metrics:
            c_val = getattr(champ_metrics, m, None)
            cand_val = getattr(cand_metrics, m, None)
            if c_val is not None and cand_val is not None:
                deltas[m] = compute_metric_delta(
                    metric_name=m,
                    champion_val=c_val,
                    candidate_val=cand_val,
                    is_rate=False,
                )

        return ClassificationComparison(
            champion_metrics=champ_metrics,
            candidate_metrics=cand_metrics,
            deltas=deltas,
        )

    def compare_operational_decisions(
        self,
        champ_op: OperationalDecisionMetrics,
        cand_op: OperationalDecisionMetrics,
    ) -> OperationalComparison:
        """
        Compute neutral mathematical deltas between Champion and Candidate operational routing.
        """
        # Action count deltas
        action_count_deltas: Dict[str, int] = {}
        action_pct_deltas: Dict[str, float] = {}
        for action in ["APPROVE", "REVIEW", "BLOCK"]:
            c_cnt = champ_op.action_counts.get(action, 0)
            cand_cnt = cand_op.action_counts.get(action, 0)
            action_count_deltas[action] = cand_cnt - c_cnt

            c_pct = champ_op.action_percentages.get(action, 0.0)
            cand_pct = cand_op.action_percentages.get(action, 0.0)
            action_pct_deltas[action] = round(cand_pct - c_pct, 4)

        # Fraud routing deltas
        fraud_routing_deltas = {
            "fraud_in_block": cand_op.fraud_in_block_count - champ_op.fraud_in_block_count,
            "fraud_in_review": cand_op.fraud_in_review_count - champ_op.fraud_in_review_count,
            "fraud_in_approve": cand_op.fraud_in_approve_count - champ_op.fraud_in_approve_count,
        }

        # Legit routing deltas
        legit_routing_deltas = {
            "legit_in_block": cand_op.legit_in_block_count - champ_op.legit_in_block_count,
            "legit_in_review": cand_op.legit_in_review_count - champ_op.legit_in_review_count,
            "legit_in_approve": cand_op.legit_in_approve_count - champ_op.legit_in_approve_count,
        }

        # Queue purity deltas
        queue_purity_deltas: Dict[str, MetricDelta] = {}
        if champ_op.block_precision is not None and cand_op.block_precision is not None:
            queue_purity_deltas["block_precision"] = compute_metric_delta(
                "block_precision",
                champ_op.block_precision,
                cand_op.block_precision,
                is_rate=True,
            )
        if champ_op.review_queue_purity is not None and cand_op.review_queue_purity is not None:
            queue_purity_deltas["review_queue_purity"] = compute_metric_delta(
                "review_queue_purity",
                champ_op.review_queue_purity,
                cand_op.review_queue_purity,
                is_rate=True,
            )

        rule_override_delta = cand_op.rule_overridden_count - champ_op.rule_overridden_count

        return OperationalComparison(
            champion_operational=champ_op,
            candidate_operational=cand_op,
            action_count_deltas=action_count_deltas,
            action_percentage_deltas=action_pct_deltas,
            fraud_routing_deltas=fraud_routing_deltas,
            legit_routing_deltas=legit_routing_deltas,
            queue_purity_deltas=queue_purity_deltas,
            rule_overridden_delta=rule_override_delta,
        )

    def compare(
        self,
        candidate_bundle_dir: Union[str, Path],
        champion_source: Optional[Union[str, Path, Dict[str, Path]]] = None,
        val_dataset_path: Union[str, Path] = VAL_FEATURES_PATH,
        oot_dataset_path: Union[str, Path] = TEST_FEATURES_PATH,
        output_comparison_file: Optional[Union[str, Path]] = None,
        benchmark_latency: bool = True,
        policy_mode: PolicyMode = PolicyMode.TRI_TIER,
        rule_engine: Optional[RuleEngine] = None,
    ) -> ChampionChallengerComparisonResult:
        """
        Execute full Champion vs. Candidate comparative evaluation.

        Guarantees:
        - Evaluates on identical validation and protected OOT splits (55 features).
        - Candidate uses frozen threshold tau* (no re-tuning).
        - Champion uses authoritative metadata threshold (dynamically loaded).
        - Protected OOT dataset is strictly read-only.
        - Verifies Champion checksums before and after to enforce immutability.
        - Strictly preserves Candidate status = CANDIDATE, is_active_champion = False.
        - Computes neutral mathematical deltas only.
        """
        # 1. Load Champion and Candidate bundles
        (
            champ_model,
            champ_prep,
            champ_meta,
            champ_threshold,
            champ_paths,
        ) = self.load_authoritative_champion(champion_source)

        (
            cand_model,
            cand_prep,
            cand_manifest,
            cand_threshold,
            cand_paths,
        ) = self.load_candidate_bundle(candidate_bundle_dir)

        # 2. Snapshot Champion checksums before evaluation
        initial_champ_checksums = self._snapshot_champion_checksums(
            model_path=champ_paths["model"],
            preprocessor_path=champ_paths["preprocessor"],
            metadata_path=champ_paths.get("metadata"),
        )

        cand_checksums = {
            "model": calculate_file_sha256(cand_paths["model"]),
            "preprocessor": calculate_file_sha256(cand_paths["preprocessor"]),
            "manifest": calculate_file_sha256(cand_paths["manifest"]),
        }

        # 3. Load and validate identical validation and OOT partitions
        val_path = Path(val_dataset_path).resolve()
        oot_path = Path(oot_dataset_path).resolve()

        if not val_path.exists():
            raise FileNotFoundError(f"Validation dataset not found at: {val_path}")
        if not oot_path.exists():
            raise FileNotFoundError(f"Protected OOT dataset not found at: {oot_path}")

        val_df = pd.read_parquet(val_path)
        oot_df = pd.read_parquet(oot_path)

        X_val, y_val = prepare_features_and_target(val_df)
        X_oot, y_oot = prepare_features_and_target(oot_df)

        if len(X_val.columns) != self.config.expected_feature_count:
            raise ValueError(
                f"Validation features count {len(X_val.columns)} != expected {self.config.expected_feature_count}."
            )
        if len(X_oot.columns) != self.config.expected_feature_count:
            raise ValueError(
                f"OOT features count {len(X_oot.columns)} != expected {self.config.expected_feature_count}."
            )

        # 4. Generate Predictions on Validation Set
        X_val_champ_trans = champ_prep.transform(X_val)
        X_val_cand_trans = cand_prep.transform(X_val)

        p_val_champ = champ_model.predict_proba(X_val_champ_trans)
        y_prob_val_champ = (
            p_val_champ[:, 1].astype(np.float64)
            if hasattr(p_val_champ, "ndim") and p_val_champ.ndim == 2 and p_val_champ.shape[1] > 1
            else np.ravel(p_val_champ).astype(np.float64)
        )

        p_val_cand = cand_model.predict_proba(X_val_cand_trans)
        y_prob_val_cand = (
            p_val_cand[:, 1].astype(np.float64)
            if hasattr(p_val_cand, "ndim") and p_val_cand.ndim == 2 and p_val_cand.shape[1] > 1
            else np.ravel(p_val_cand).astype(np.float64)
        )

        champ_val_metrics = calculate_evaluation_metrics(
            y_true=y_val,
            y_prob=y_prob_val_champ,
            threshold=champ_threshold,
            cost_config=self.cost_config,
        )

        cand_val_metrics = calculate_evaluation_metrics(
            y_true=y_val,
            y_prob=y_prob_val_cand,
            threshold=cand_threshold,
            cost_config=self.cost_config,
        )

        val_comparison = self.compare_classification_metrics(champ_val_metrics, cand_val_metrics)

        # 5. Generate Predictions on Protected OOT Set (Strictly Read-Only)
        X_oot_champ_trans = champ_prep.transform(X_oot)
        X_oot_cand_trans = cand_prep.transform(X_oot)

        p_oot_champ = champ_model.predict_proba(X_oot_champ_trans)
        y_prob_oot_champ = (
            p_oot_champ[:, 1].astype(np.float64)
            if hasattr(p_oot_champ, "ndim") and p_oot_champ.ndim == 2 and p_oot_champ.shape[1] > 1
            else np.ravel(p_oot_champ).astype(np.float64)
        )

        p_oot_cand = cand_model.predict_proba(X_oot_cand_trans)
        y_prob_oot_cand = (
            p_oot_cand[:, 1].astype(np.float64)
            if hasattr(p_oot_cand, "ndim") and p_oot_cand.ndim == 2 and p_oot_cand.shape[1] > 1
            else np.ravel(p_oot_cand).astype(np.float64)
        )

        champ_oot_metrics = calculate_evaluation_metrics(
            y_true=y_oot,
            y_prob=y_prob_oot_champ,
            threshold=champ_threshold,
            cost_config=self.cost_config,
        )

        cand_oot_metrics = calculate_evaluation_metrics(
            y_true=y_oot,
            y_prob=y_prob_oot_cand,
            threshold=cand_threshold,
            cost_config=self.cost_config,
        )

        oot_comparison = self.compare_classification_metrics(champ_oot_metrics, cand_oot_metrics)

        # 6. Operational Routing Comparison
        champ_val_op = self.operational_evaluator.evaluate_operational_decisions(
            df=val_df,
            model_path=champ_paths["model"],
            preprocessor_path=champ_paths["preprocessor"],
            operating_threshold=champ_threshold,
            metadata_path=champ_paths.get("metadata"),
            policy_mode=policy_mode,
            rule_engine=rule_engine,
        )

        cand_val_op = self.operational_evaluator.evaluate_operational_decisions(
            df=val_df,
            model_path=cand_paths["model"],
            preprocessor_path=cand_paths["preprocessor"],
            operating_threshold=cand_threshold,
            metadata_path=cand_paths.get("manifest"),
            policy_mode=policy_mode,
            rule_engine=rule_engine,
        )

        val_op_comparison = self.compare_operational_decisions(champ_val_op, cand_val_op)

        champ_oot_op = self.operational_evaluator.evaluate_operational_decisions(
            df=oot_df,
            model_path=champ_paths["model"],
            preprocessor_path=champ_paths["preprocessor"],
            operating_threshold=champ_threshold,
            metadata_path=champ_paths.get("metadata"),
            policy_mode=policy_mode,
            rule_engine=rule_engine,
        )

        cand_oot_op = self.operational_evaluator.evaluate_operational_decisions(
            df=oot_df,
            model_path=cand_paths["model"],
            preprocessor_path=cand_paths["preprocessor"],
            operating_threshold=cand_threshold,
            metadata_path=cand_paths.get("manifest"),
            policy_mode=policy_mode,
            rule_engine=rule_engine,
        )

        oot_op_comparison = self.compare_operational_decisions(champ_oot_op, cand_oot_op)

        # 7. Standardized Latency Benchmarking
        if benchmark_latency:
            lat_comparison = self.benchmark_identical_latency(
                champ_model=champ_model,
                champ_prep=champ_prep,
                cand_model=cand_model,
                cand_prep=cand_prep,
                X_val=X_val,
                sample_count=50,
                warmup_count=5,
            )
        else:
            dummy_stats = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
            lat_comparison = LatencyComparison(
                champion_latency=dummy_stats,
                candidate_latency=dummy_stats,
                deltas={
                    k: compute_metric_delta(k, 0.0, 0.0, is_rate=False)
                    for k in dummy_stats
                },
                warmup_iterations=0,
                measured_samples=0,
            )

        # 8. Assert Champion Immutability after all evaluations
        self._verify_champion_immutability(
            initial_checksums=initial_champ_checksums,
            model_path=champ_paths["model"],
            preprocessor_path=champ_paths["preprocessor"],
            metadata_path=champ_paths.get("metadata"),
        )

        # 9. Verify Candidate State Invariants
        if cand_manifest.status != ModelLifecycleStatus.CANDIDATE:
            raise ValueError(
                f"Candidate lifecycle status corrupted! Expected CANDIDATE, found {cand_manifest.status}"
            )

        champ_version = champ_meta.get("model_version", "1.0.0")
        cand_version = cand_manifest.model_version

        dataset_meta = {
            "validation_sample_count": len(val_df),
            "validation_fraud_count": int(np.sum(y_val == 1)),
            "validation_legit_count": int(np.sum(y_val == 0)),
            "oot_sample_count": len(oot_df),
            "oot_fraud_count": int(np.sum(y_oot == 1)),
            "oot_legit_count": int(np.sum(y_oot == 0)),
            "feature_count": len(X_val.columns),
        }

        # 10. Construct self-contained result
        comparison_result = ChampionChallengerComparisonResult(
            champion_version=str(champ_version),
            candidate_version=str(cand_version),
            champion_operating_threshold=champ_threshold,
            candidate_operating_threshold=cand_threshold,
            validation_comparison=val_comparison,
            oot_comparison=oot_comparison,
            validation_operational_comparison=val_op_comparison,
            oot_operational_comparison=oot_op_comparison,
            latency_comparison=lat_comparison,
            dataset_metadata=dataset_meta,
            champion_sha256_checksums=initial_champ_checksums,
            candidate_sha256_checksums=cand_checksums,
        )

        # 11. Persist result artifact
        if output_comparison_file is not None:
            out_file = Path(output_comparison_file).resolve()
        else:
            comp_dir = self.config.registry_dir / "comparisons"
            comp_dir.mkdir(parents=True, exist_ok=True)
            out_file = comp_dir / f"comparison_v{champ_version}_vs_v{cand_version}.json"

        comparison_result.save(out_file)
        logger.info(f"Comparison artifact saved to: {out_file}")

        return comparison_result
