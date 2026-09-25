"""
Deterministic Promotion Gate Evaluator for Phase 14.5.

Consumes the Phase 14.4 Champion vs. Candidate comparative evaluation artifact,
performs dynamic cryptographic verification of on-disk assets, and applies multi-dimensional
governance criteria to determine promotion eligibility.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ml.lifecycle.config import LifecycleConfig, default_lifecycle_config
from ml.lifecycle.evaluation.comparison_schemas import ChampionChallengerComparisonResult
from ml.lifecycle.governance.config import (
    PromotionGateSpecification,
    default_promotion_gate_specification,
)
from ml.lifecycle.governance.schemas import (
    GateStatus,
    PromotionEligibilityAssessment,
    PromotionGateResult,
)
from ml.lifecycle.schemas import calculate_file_sha256


class EvidenceCorruptedError(ValueError):
    """Raised when comparison evidence is missing, invalid, or corrupted."""
    pass


class PromotionGateEvaluator:
    """
    Deterministic governance engine evaluating Candidate promotion eligibility.

    Guarantees:
    - Zero model retraining, tuning, or re-evaluation.
    - Consumes verified Phase 14.4 comparison artifact as primary evidence.
    - Dynamically computes and verifies SHA-256 hashes for all physical assets.
    - Completely auditable, transparent gate-by-gate decision matrix.
    """

    def __init__(
        self,
        specification: Optional[PromotionGateSpecification] = None,
        lifecycle_config: Optional[LifecycleConfig] = None,
    ) -> None:
        """
        Initialize the Promotion Gate Evaluator.

        Args:
            specification: Governance policy criteria (defaults to standard specification).
            lifecycle_config: Model lifecycle path configuration.
        """
        self.spec = specification or default_promotion_gate_specification
        self.lifecycle_cfg = lifecycle_config or default_lifecycle_config

    def evaluate_eligibility(
        self,
        comparison_evidence: Union[str, Path, ChampionChallengerComparisonResult, Dict[str, Any]],
        champion_dir: Optional[Union[str, Path]] = None,
    ) -> PromotionEligibilityAssessment:
        """
        Evaluate candidate promotion eligibility against the full governance specification.

        Args:
            comparison_evidence: Path to comparison JSON file, comparison result model, or dict.
            champion_dir: Optional override path to Champion artifacts directory.

        Returns:
            PromotionEligibilityAssessment containing all gate results and eligibility status.
        """
        evidence_path_str: str = "in_memory"
        evidence_sha256: str = "in_memory"
        comparison: ChampionChallengerComparisonResult

        if isinstance(comparison_evidence, (str, Path)):
            p = Path(comparison_evidence).resolve()
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Comparison evidence artifact not found at: {p}")
            evidence_path_str = str(p)
            evidence_sha256 = calculate_file_sha256(p)
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                comparison = ChampionChallengerComparisonResult.model_validate(data)
            except Exception as e:
                raise EvidenceCorruptedError(f"Failed to parse comparison evidence artifact: {e}") from e
        elif isinstance(comparison_evidence, ChampionChallengerComparisonResult):
            comparison = comparison_evidence
            dumped = json.dumps(comparison.model_dump(mode="json"), sort_keys=True).encode("utf-8")
            import hashlib
            evidence_sha256 = hashlib.sha256(dumped).hexdigest()
        elif isinstance(comparison_evidence, dict):
            try:
                comparison = ChampionChallengerComparisonResult.model_validate(comparison_evidence)
                dumped = json.dumps(comparison_evidence, sort_keys=True).encode("utf-8")
                import hashlib
                evidence_sha256 = hashlib.sha256(dumped).hexdigest()
            except Exception as e:
                raise EvidenceCorruptedError(f"Invalid comparison evidence dictionary: {e}") from e
        else:
            raise TypeError(f"Unsupported comparison evidence type: {type(comparison_evidence).__name__}")

        champ_dir_p = Path(champion_dir).resolve() if champion_dir else self.lifecycle_cfg.champion_model_path.parent

        # Execute all 11 individual gate evaluations
        gate_results: List[PromotionGateResult] = []

        # 1. OOT PR-AUC Parity Gate (Governance Policy)
        gate_results.append(self._eval_oot_pr_auc_parity(comparison))

        # 2. Validation PR-AUC Parity Gate (Governance Policy)
        gate_results.append(self._eval_val_pr_auc_parity(comparison))

        # 3. OOT Fraud Catch Rate Non-Regression Gate (Governance Policy)
        gate_results.append(self._eval_oot_recall_non_regression(comparison))

        # 4. OOT Missed Fraud Non-Increase Gate (Governance Policy)
        gate_results.append(self._eval_oot_fn_non_increase(comparison))

        # 5. OOT Financial Cost Non-Increase Gate (Authoritative Baseline + Governance Policy)
        gate_results.append(self._eval_oot_cost_non_increase(comparison))

        # 6. Operational False Positive Rate Ceiling Gate (Governance Policy)
        gate_results.append(self._eval_oot_fpr_ceiling(comparison))

        # 7. Automated Block Precision Floor Gate (Governance Policy)
        gate_results.append(self._eval_oot_block_precision_floor(comparison))

        # 8. Production Inference Latency SLA Gate (Governance Policy)
        gate_results.append(self._eval_latency_sla(comparison))

        # 9. Dynamic Champion Artifact Integrity Gate (Existing System Requirement)
        gate_results.append(self._eval_dynamic_champion_integrity(comparison, champ_dir_p))

        # 10A. Predictive Feature Schema Compatibility Gate (Existing System Requirement)
        gate_results.append(self._eval_feature_schema_compatibility(comparison))

        # 10B. Evaluation Dataset Partition Integrity Gate (Existing System Requirement)
        gate_results.append(self._eval_partition_integrity(comparison))

        # Compute summary metrics
        total_gates = len(gate_results)
        passed_count = sum(1 for g in gate_results if g.status == GateStatus.PASS)
        failed_count = sum(1 for g in gate_results if g.status == GateStatus.FAIL)
        blocked_count = sum(1 for g in gate_results if g.status == GateStatus.BLOCKED)

        is_eligible = (failed_count == 0 and blocked_count == 0)

        now_utc = datetime.now(timezone.utc).isoformat()

        return PromotionEligibilityAssessment(
            candidate_version=comparison.candidate_version,
            champion_version=comparison.champion_version,
            is_eligible=is_eligible,
            total_gates_evaluated=total_gates,
            passed_gates_count=passed_count,
            failed_gates_count=failed_count,
            blocked_gates_count=blocked_count,
            gate_results=gate_results,
            evidence_artifact_path=evidence_path_str,
            evidence_artifact_sha256=evidence_sha256,
            evaluated_at=now_utc,
            assessed_by="promotion_gate_evaluator",
        )

    # -------------------------------------------------------------------------
    # Individual Gate Evaluation Methods
    # -------------------------------------------------------------------------

    def _eval_oot_pr_auc_parity(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        champ_val = comp.oot_comparison.champion_metrics.pr_auc
        cand_val = comp.oot_comparison.candidate_metrics.pr_auc
        delta = round(cand_val - champ_val, 5)
        tol = self.spec.max_allowed_oot_pr_auc_degradation
        req_val = round(champ_val - tol, 5)
        passed = cand_val >= req_val

        return PromotionGateResult(
            gate_id="gate_oot_pr_auc_parity",
            category="Statistical Ranking",
            metric_name="oot_pr_auc",
            operator=">=",
            required_value=req_val,
            actual_value=cand_val,
            champion_value=champ_val,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description=f"Candidate OOT PR-AUC must maintain statistical parity with Champion (tolerance margin: {tol:.4f}).",
            failure_reason=None if passed else f"Candidate OOT PR-AUC ({cand_val:.5f}) below required threshold ({req_val:.5f}).",
        )

    def _eval_val_pr_auc_parity(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        champ_val = comp.validation_comparison.champion_metrics.pr_auc
        cand_val = comp.validation_comparison.candidate_metrics.pr_auc
        delta = round(cand_val - champ_val, 5)
        tol = self.spec.max_allowed_val_pr_auc_degradation
        req_val = round(champ_val - tol, 5)
        passed = cand_val >= req_val

        return PromotionGateResult(
            gate_id="gate_val_pr_auc_parity",
            category="Statistical Ranking",
            metric_name="validation_pr_auc",
            operator=">=",
            required_value=req_val,
            actual_value=cand_val,
            champion_value=champ_val,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description=f"Candidate Validation PR-AUC must maintain statistical parity with Champion (tolerance margin: {tol:.4f}).",
            failure_reason=None if passed else f"Candidate Validation PR-AUC ({cand_val:.5f}) below required threshold ({req_val:.5f}).",
        )

    def _eval_oot_recall_non_regression(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        champ_val = comp.oot_comparison.champion_metrics.recall
        cand_val = comp.oot_comparison.candidate_metrics.recall
        delta = round(cand_val - champ_val, 5)
        req_val = champ_val if self.spec.require_oot_recall_non_regression else 0.0
        passed = cand_val >= req_val

        return PromotionGateResult(
            gate_id="gate_oot_recall_non_regression",
            category="Detection Effectiveness",
            metric_name="oot_recall",
            operator=">=",
            required_value=req_val,
            actual_value=cand_val,
            champion_value=champ_val,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description="Candidate OOT Fraud Catch Rate (Recall) must not regress below Champion baseline.",
            failure_reason=None if passed else f"Candidate OOT Recall ({cand_val:.5f}) regressed below Champion ({champ_val:.5f}).",
        )

    def _eval_oot_fn_non_increase(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        champ_val = comp.oot_comparison.champion_metrics.fn
        cand_val = comp.oot_comparison.candidate_metrics.fn
        if champ_val is None or cand_val is None:
            return PromotionGateResult(
                gate_id="gate_oot_fn_non_increase",
                category="Loss Mitigation",
                metric_name="oot_false_negatives",
                operator="<=",
                required_value="non_null",
                actual_value="null",
                champion_value=champ_val,
                delta_value=None,
                status=GateStatus.BLOCKED,
                is_policy_configuration=True,
                description="Candidate OOT False Negatives must not increase above Champion count.",
                failure_reason="FN count is missing from evaluation evidence.",
            )

        delta = cand_val - champ_val
        passed = cand_val <= champ_val if self.spec.require_oot_fn_non_increase else True

        return PromotionGateResult(
            gate_id="gate_oot_fn_non_increase",
            category="Loss Mitigation",
            metric_name="oot_false_negatives",
            operator="<=",
            required_value=champ_val,
            actual_value=cand_val,
            champion_value=champ_val,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description="Candidate OOT False Negatives (missed frauds) must not increase above Champion count.",
            failure_reason=None if passed else f"Candidate OOT False Negatives ({cand_val}) exceeded Champion ({champ_val}).",
        )

    def _eval_oot_cost_non_increase(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        champ_val = comp.oot_comparison.champion_metrics.expected_cost
        cand_val = comp.oot_comparison.candidate_metrics.expected_cost
        if champ_val is None or cand_val is None:
            return PromotionGateResult(
                gate_id="gate_oot_cost_non_increase",
                category="Financial Cost",
                metric_name="oot_expected_cost",
                operator="<=",
                required_value="non_null",
                actual_value="null",
                champion_value=champ_val,
                delta_value=None,
                status=GateStatus.BLOCKED,
                is_policy_configuration=True,
                description="Candidate OOT Expected Financial Cost must not exceed Champion Cost under authoritative CostConfig.",
                failure_reason="Expected cost is missing from evaluation evidence.",
            )

        delta = round(cand_val - champ_val, 2)
        passed = cand_val <= champ_val if self.spec.require_oot_cost_non_increase else True

        return PromotionGateResult(
            gate_id="gate_oot_cost_non_increase",
            category="Financial Cost",
            metric_name="oot_expected_cost",
            operator="<=",
            required_value=champ_val,
            actual_value=cand_val,
            champion_value=champ_val,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description="Candidate total expected decision cost on OOT holdout must not exceed Champion cost under authoritative CostConfig ($15 FP, $200 FN).",
            failure_reason=None if passed else f"Candidate OOT Expected Cost (${cand_val:,.2f}) exceeded Champion (${champ_val:,.2f}).",
        )

    def _eval_oot_fpr_ceiling(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        cand_val = comp.oot_comparison.candidate_metrics.fpr
        champ_val = comp.oot_comparison.champion_metrics.fpr
        delta = round(cand_val - champ_val, 5)
        req_val = self.spec.max_oot_fpr_ceiling
        passed = cand_val <= req_val

        return PromotionGateResult(
            gate_id="gate_oot_fpr_ceiling",
            category="Operational Safety",
            metric_name="oot_fpr",
            operator="<=",
            required_value=req_val,
            actual_value=cand_val,
            champion_value=champ_val,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description=f"Candidate OOT False Positive Rate must not exceed operational safety ceiling ({req_val:.4f}).",
            failure_reason=None if passed else f"Candidate OOT FPR ({cand_val:.5f}) exceeded safety ceiling ({req_val:.5f}).",
        )

    def _eval_oot_block_precision_floor(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        cand_val = comp.oot_comparison.candidate_metrics.precision
        champ_val = comp.oot_comparison.champion_metrics.precision
        delta = round(cand_val - champ_val, 5)
        req_val = self.spec.min_oot_block_precision
        passed = cand_val >= req_val

        return PromotionGateResult(
            gate_id="gate_oot_block_precision_floor",
            category="Decision Conviction",
            metric_name="oot_block_precision",
            operator=">=",
            required_value=req_val,
            actual_value=cand_val,
            champion_value=champ_val,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description=f"Candidate OOT Precision at operating threshold must meet or exceed minimum block conviction floor ({req_val:.4f}).",
            failure_reason=None if passed else f"Candidate OOT Precision ({cand_val:.5f}) below required floor ({req_val:.5f}).",
        )

    def _eval_latency_sla(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        cand_lat = comp.latency_comparison.candidate_latency
        champ_lat = comp.latency_comparison.champion_latency
        cand_p95 = cand_lat.get("p95_ms", 0.0) if isinstance(cand_lat, dict) else getattr(cand_lat, "p95_ms", 0.0)
        champ_p95 = champ_lat.get("p95_ms", 0.0) if isinstance(champ_lat, dict) else getattr(champ_lat, "p95_ms", 0.0)
        delta = round(cand_p95 - champ_p95, 3)
        max_abs = self.spec.max_p95_latency_ms
        max_ratio = self.spec.max_latency_ratio_vs_champion
        ratio_limit = round(champ_p95 * max_ratio, 3)
        req_val = min(max_abs, ratio_limit)

        passed = (cand_p95 <= max_abs) and (cand_p95 <= ratio_limit)

        return PromotionGateResult(
            gate_id="gate_latency_p95_sla",
            category="Inference SLA",
            metric_name="p95_latency_ms",
            operator="<=",
            required_value=req_val,
            actual_value=cand_p95,
            champion_value=champ_p95,
            delta_value=delta,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=True,
            description=f"Candidate p95 latency must satisfy production SLA ceiling ({max_abs:.2f} ms) and not exceed {max_ratio:.2f}x Champion p95.",
            failure_reason=None if passed else f"Candidate p95 latency ({cand_p95:.3f} ms) exceeded allowed limit ({req_val:.3f} ms).",
        )

    def _eval_dynamic_champion_integrity(
        self,
        comp: ChampionChallengerComparisonResult,
        champ_dir: Path,
    ) -> PromotionGateResult:
        if not self.spec.require_dynamic_champion_integrity:
            return PromotionGateResult(
                gate_id="gate_champion_dynamic_integrity",
                category="Cryptographic Integrity",
                metric_name="champion_checksum_match",
                operator="==",
                required_value=True,
                actual_value=True,
                champion_value=None,
                delta_value=None,
                status=GateStatus.PASS,
                is_policy_configuration=False,
                description="Dynamic Champion asset integrity verification (skipped by config).",
                failure_reason=None,
            )

        # Compute dynamic on-disk hashes
        model_file = champ_dir / "champion_model.joblib"
        prep_file = champ_dir / "champion_preprocessor.joblib"
        meta_file = champ_dir / "model_metadata.json"

        missing = []
        for f in [model_file, prep_file, meta_file]:
            if not f.exists():
                missing.append(f.name)

        if missing:
            return PromotionGateResult(
                gate_id="gate_champion_dynamic_integrity",
                category="Cryptographic Integrity",
                metric_name="champion_checksum_match",
                operator="==",
                required_value="all_files_present",
                actual_value=f"missing: {missing}",
                champion_value=None,
                delta_value=None,
                status=GateStatus.BLOCKED,
                is_policy_configuration=False,
                description="Dynamic calculation of on-disk Champion SHA-256 hashes matching registered metadata.",
                failure_reason=f"Champion physical assets missing on disk: {missing}",
            )

        dynamic_m_sha = calculate_file_sha256(model_file)
        dynamic_p_sha = calculate_file_sha256(prep_file)
        dynamic_meta_sha = calculate_file_sha256(meta_file)

        # Compare against registered checksums in evidence
        reg_m = comp.champion_sha256_checksums.get("model", "").lower()
        reg_p = comp.champion_sha256_checksums.get("preprocessor", "").lower()
        reg_meta = comp.champion_sha256_checksums.get("metadata", "").lower()

        matches = (
            dynamic_m_sha.lower() == reg_m
            and dynamic_p_sha.lower() == reg_p
            and dynamic_meta_sha.lower() == reg_meta
        )

        return PromotionGateResult(
            gate_id="gate_champion_dynamic_integrity",
            category="Cryptographic Integrity",
            metric_name="champion_checksum_match",
            operator="==",
            required_value="registered_hashes_match",
            actual_value="hashes_match" if matches else "mismatch_detected",
            champion_value=None,
            delta_value=None,
            status=GateStatus.PASS if matches else GateStatus.FAIL,
            is_policy_configuration=False,
            description="Dynamic calculation of on-disk Champion SHA-256 hashes matching registered bundle checksums.",
            failure_reason=None if matches else "Dynamic Champion checksums on disk do not match registered metadata hashes.",
        )

    def _eval_feature_schema_compatibility(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        actual_count = comp.dataset_metadata.get("feature_count", 0)
        req_count = self.spec.expected_feature_count
        passed = (actual_count == req_count)

        return PromotionGateResult(
            gate_id="gate_feature_schema_compatibility",
            category="Schema Integrity",
            metric_name="feature_count",
            operator="==",
            required_value=req_count,
            actual_value=actual_count,
            champion_value=req_count,
            delta_value=actual_count - req_count,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=False,
            description=f"Candidate feature schema must strictly match the canonical {req_count}-feature contract.",
            failure_reason=None if passed else f"Feature count ({actual_count}) does not match required schema count ({req_count}).",
        )

    def _eval_partition_integrity(self, comp: ChampionChallengerComparisonResult) -> PromotionGateResult:
        actual_val = comp.dataset_metadata.get("validation_sample_count", 0)
        actual_oot = comp.dataset_metadata.get("oot_sample_count", 0)
        req_val = self.spec.expected_validation_sample_count
        req_oot = self.spec.expected_oot_sample_count

        passed = (actual_val == req_val and actual_oot == req_oot)

        return PromotionGateResult(
            gate_id="gate_evaluation_partition_integrity",
            category="Partition Integrity",
            metric_name="evaluation_sample_counts",
            operator="==",
            required_value=f"val={req_val}, oot={req_oot}",
            actual_value=f"val={actual_val}, oot={actual_oot}",
            champion_value=f"val={req_val}, oot={req_oot}",
            delta_value=0 if passed else -1,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            is_policy_configuration=False,
            description="Evaluation datasets must exactly match frozen Validation and protected OOT partition sizes.",
            failure_reason=None if passed else f"Partition row count mismatch: val={actual_val} (req {req_val}), oot={actual_oot} (req {req_oot}).",
        )
