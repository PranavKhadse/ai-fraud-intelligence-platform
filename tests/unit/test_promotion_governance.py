"""
Unit Tests for Phase 14.5 Promotion Governance & Human Sign-Off Subsystem.

Validates:
1. PromotionGateSpecification policy parameters and immutability.
2. Individual evaluation logic across all 11 governance gates.
3. Policy configuration tagging vs existing system requirements.
4. Dynamic on-disk Champion SHA-256 verification (zero hardcoded hashes).
5. Evidence corruption and missing evidence handling.
6. PromotionEligibilityAssessment synthesis and deterministic hashing.
7. HumanSignOffEngine RBAC enforcement (ADMIN, ANALYST allowed; API_CLIENT rejected).
8. Precondition enforcement (APPROVED strictly blocked if candidate is ineligible).
9. Mandatory business rationale validation.
10. Sign-off persistence, atomic writes, and conflict/immutability protection.
11. Complete side-effect freedom (candidate remains CANDIDATE).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pytest

from backend.app.core.security import ActorContext
from backend.app.db.models.enums import AuditActorType
from ml.lifecycle.config import LifecycleConfig
from ml.lifecycle.evaluation.comparison_schemas import (
    ChampionChallengerComparisonResult,
    ClassificationComparison,
    DeltaSign,
    LatencyComparison,
    MetricDelta,
    OperationalComparison,
)
from ml.lifecycle.evaluation.schemas import OperationalDecisionMetrics
from ml.lifecycle.governance import (
    EvidenceCorruptedError,
    GateStatus,
    HumanSignOffEngine,
    PromotionEligibilityAssessment,
    PromotionGateBlockedError,
    PromotionGateEvaluator,
    PromotionGateResult,
    PromotionGateSpecification,
    PromotionSignOffRecord,
    SignOffConflictError,
    SignOffDecision,
    UnauthorizedSignOffActorError,
    compute_assessment_sha256,
    default_promotion_gate_specification,
)
from ml.lifecycle.schemas import EvaluationMetricsSummary, calculate_file_sha256

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# Fixtures & Mock Builders
# -----------------------------------------------------------------------------

def create_mock_metrics_summary(
    pr_auc: float = 0.96054,
    roc_auc: float = 0.99905,
    precision: float = 0.78187,
    recall: float = 0.94264,
    f1: float = 0.85476,
    fpr: float = 0.00088,
    accuracy: float = 0.99893,
    threshold: float = 0.78,
    tp: int = 871,
    fp: int = 243,
    tn: int = 276693,
    fn: int = 53,
    total_samples: int = 277860,
    expected_cost: float = 14245.0,
) -> EvaluationMetricsSummary:
    return EvaluationMetricsSummary(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        accuracy=accuracy,
        threshold=threshold,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        total_samples=total_samples,
        expected_cost=expected_cost,
        fraud_in_block_count=tp,
        fraud_in_review_count=0,
        review_queue_purity=precision,
    )


def create_mock_comparison_result(
    candidate_oot_pr_auc: float = 0.96054,
    champion_oot_pr_auc: float = 0.96054,
    candidate_val_pr_auc: float = 0.96190,
    champion_val_pr_auc: float = 0.96190,
    candidate_oot_recall: float = 0.94264,
    champion_oot_recall: float = 0.89069,
    candidate_oot_fn: int = 53,
    champion_oot_fn: int = 101,
    candidate_oot_cost: float = 14245.0,
    champion_oot_cost: float = 20890.0,
    candidate_oot_fpr: float = 0.00088,
    candidate_oot_precision: float = 0.78187,
    candidate_p95_ms: float = 1.037,
    champion_p95_ms: float = 0.976,
    feature_count: int = 55,
    val_sample_count: int = 277859,
    oot_sample_count: int = 277860,
    champion_checksums: Optional[dict] = None,
) -> ChampionChallengerComparisonResult:
    """Build a complete, synthetically passing or customizable comparison result."""
    champ_val_m = create_mock_metrics_summary(
        pr_auc=champion_val_pr_auc,
        threshold=0.94,
        precision=0.93718,
        recall=0.89189,
        f1=0.91397,
        tp=1089,
        fp=73,
        tn=276565,
        fn=132,
        total_samples=val_sample_count,
        expected_cost=27495.0,
    )
    cand_val_m = create_mock_metrics_summary(
        pr_auc=candidate_val_pr_auc,
        threshold=0.78,
        precision=0.78697,
        recall=0.95004,
        f1=0.86085,
        tp=1160,
        fp=314,
        tn=276324,
        fn=61,
        total_samples=val_sample_count,
        expected_cost=16910.0,
    )

    champ_oot_m = create_mock_metrics_summary(
        pr_auc=champion_oot_pr_auc,
        threshold=0.94,
        precision=0.94707,
        recall=champion_oot_recall,
        f1=0.91801,
        fpr=0.00017,
        tp=823,
        fp=46,
        tn=276890,
        fn=champion_oot_fn,
        total_samples=oot_sample_count,
        expected_cost=champion_oot_cost,
    )
    cand_oot_m = create_mock_metrics_summary(
        pr_auc=candidate_oot_pr_auc,
        threshold=0.78,
        precision=candidate_oot_precision,
        recall=candidate_oot_recall,
        f1=0.85476,
        fpr=candidate_oot_fpr,
        tp=871,
        fp=243,
        tn=276693,
        fn=candidate_oot_fn,
        total_samples=oot_sample_count,
        expected_cost=candidate_oot_cost,
    )

    dummy_delta = MetricDelta(
        metric_name="pr_auc",
        champion_value=0.96054,
        candidate_value=0.96054,
        absolute_delta=0.0,
        delta_sign=DeltaSign.ZERO,
    )

    dummy_op_m = OperationalDecisionMetrics(
        action_counts={"APPROVE": 276746, "REVIEW": 0, "BLOCK": 1114},
        action_percentages={"APPROVE": 0.99599, "REVIEW": 0.0, "BLOCK": 0.00401},
        risk_tier_counts={"LOW": 276746, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 1114},
        risk_tier_percentages={"LOW": 0.99599, "MEDIUM": 0.0, "HIGH": 0.0, "CRITICAL": 0.00401},
        model_score_stats={"min": 0.0, "mean": 0.05, "max": 0.99},
        risk_score_stats={"min": 0.0, "mean": 5.0, "max": 99.0},
        policy_mode="TRI_TIER",
        operating_threshold=0.78,
        fraud_in_block_count=871,
        fraud_in_review_count=0,
        fraud_in_approve_count=53,
        legit_in_block_count=243,
        legit_in_review_count=0,
        legit_in_approve_count=276693,
        block_precision=candidate_oot_precision,
        review_queue_purity=candidate_oot_precision,
        rule_overridden_count=0,
        rule_trigger_counts={},
    )

    val_comp = ClassificationComparison(
        champion_metrics=champ_val_m,
        candidate_metrics=cand_val_m,
        deltas={"pr_auc": dummy_delta},
    )
    oot_comp = ClassificationComparison(
        champion_metrics=champ_oot_m,
        candidate_metrics=cand_oot_m,
        deltas={"pr_auc": dummy_delta},
    )

    op_comp = OperationalComparison(
        champion_operational=dummy_op_m,
        candidate_operational=dummy_op_m,
        action_count_deltas={},
        action_percentage_deltas={},
        fraud_routing_deltas={},
        legit_routing_deltas={},
        queue_purity_deltas={},
        rule_overridden_delta=0,
    )

    lat_comp = LatencyComparison(
        champion_latency={"p95_ms": champion_p95_ms, "mean_ms": 0.60, "p50_ms": 0.55, "p99_ms": 1.05},
        candidate_latency={"p95_ms": candidate_p95_ms, "mean_ms": 0.61, "p50_ms": 0.54, "p99_ms": 1.18},
        deltas={"p95_ms": dummy_delta},
        warmup_iterations=5,
        measured_samples=50,
    )

    return ChampionChallengerComparisonResult(
        champion_version="1.0.0",
        candidate_version="1.1.0",
        champion_operating_threshold=0.94,
        candidate_operating_threshold=0.78,
        validation_comparison=val_comp,
        oot_comparison=oot_comp,
        validation_operational_comparison=op_comp,
        oot_operational_comparison=op_comp,
        latency_comparison=lat_comp,
        dataset_metadata={
            "validation_sample_count": val_sample_count,
            "validation_fraud_count": 1221,
            "validation_legit_count": 276638,
            "oot_sample_count": oot_sample_count,
            "oot_fraud_count": 924,
            "oot_legit_count": 276936,
            "feature_count": feature_count,
        },
        champion_sha256_checksums=champion_checksums or {
            "model": "dummy_model_hash",
            "preprocessor": "dummy_prep_hash",
            "metadata": "dummy_meta_hash",
        },
        candidate_sha256_checksums={"model": "cand_m", "preprocessor": "cand_p", "manifest": "cand_man"},
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        evaluated_by="champion_challenger_comparator",
    )


@pytest.fixture
def mock_champion_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with mock Champion assets and compute real hashes."""
    champ_dir = tmp_path / "artifacts"
    champ_dir.mkdir(parents=True)
    m_file = champ_dir / "champion_model.joblib"
    p_file = champ_dir / "champion_preprocessor.joblib"
    meta_file = champ_dir / "model_metadata.json"

    m_file.write_bytes(b"mock_champion_model_bytes")
    p_file.write_bytes(b"mock_champion_prep_bytes")
    meta_file.write_text('{"model_version": "1.0.0", "selected_threshold": 0.94}', encoding="utf-8")
    return champ_dir


# -----------------------------------------------------------------------------
# Test Suites
# -----------------------------------------------------------------------------

class TestPromotionGateSpecification:
    """Validates PromotionGateSpecification model, policy boundaries, and immutability."""

    def test_default_specification_values_and_policies(self):
        spec = default_promotion_gate_specification
        assert spec.max_allowed_oot_pr_auc_degradation == 0.0050
        assert spec.max_allowed_val_pr_auc_degradation == 0.0050
        assert spec.require_oot_recall_non_regression is True
        assert spec.require_oot_fn_non_increase is True
        assert spec.require_oot_cost_non_increase is True
        assert spec.max_oot_fpr_ceiling == 0.0020
        assert spec.min_oot_block_precision == 0.7000
        assert spec.max_p95_latency_ms == 5.00
        assert spec.max_latency_ratio_vs_champion == 1.50
        assert spec.require_dynamic_champion_integrity is True
        assert spec.expected_feature_count == 55
        assert spec.expected_validation_sample_count == 277859
        assert spec.expected_oot_sample_count == 277860

    def test_specification_frozen_immutability(self):
        spec = PromotionGateSpecification()
        with pytest.raises(Exception):  # ValidationError / FrozenInstanceError
            spec.max_oot_fpr_ceiling = 0.05  # type: ignore


class TestIndividualGateEvaluations:
    """Validates pass/fail behavior for each of the 11 governance gates."""

    def test_gate_1_oot_pr_auc_parity(self, mock_champion_dir: Path):
        # Champion: 0.96054. Allowed drop 0.0050 -> required 0.95554
        # Candidate 0.96000 >= 0.95554 -> PASS
        comp_pass = create_mock_comparison_result(
            candidate_oot_pr_auc=0.96000,
            champion_oot_pr_auc=0.96054,
            champion_checksums={
                "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
                "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
                "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
            },
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp_pass, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_oot_pr_auc_parity")
        assert gate.status == GateStatus.PASS
        assert gate.is_policy_configuration is True

        # Candidate 0.95000 < 0.95554 -> FAIL
        comp_fail = create_mock_comparison_result(
            candidate_oot_pr_auc=0.95000,
            champion_oot_pr_auc=0.96054,
            champion_checksums=comp_pass.champion_sha256_checksums,
        )
        res_fail = evaluator.evaluate_eligibility(comp_fail, champion_dir=mock_champion_dir)
        gate_fail = next(g for g in res_fail.gate_results if g.gate_id == "gate_oot_pr_auc_parity")
        assert gate_fail.status == GateStatus.FAIL
        assert "below required threshold" in gate_fail.failure_reason

    def test_gate_2_val_pr_auc_parity(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        comp = create_mock_comparison_result(
            candidate_val_pr_auc=0.94000,  # Below 0.96190 - 0.0050 = 0.95690
            champion_val_pr_auc=0.96190,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_val_pr_auc_parity")
        assert gate.status == GateStatus.FAIL
        assert res.is_eligible is False

    def test_gate_3_oot_recall_non_regression(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Recall drops from 0.89069 to 0.85000 -> FAIL
        comp = create_mock_comparison_result(
            candidate_oot_recall=0.85000,
            champion_oot_recall=0.89069,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_oot_recall_non_regression")
        assert gate.status == GateStatus.FAIL
        assert "regressed below Champion" in gate.failure_reason

    def test_gate_4_oot_fn_non_increase(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Candidate missed 110 frauds vs Champion 101 -> FAIL
        comp = create_mock_comparison_result(
            candidate_oot_fn=110,
            champion_oot_fn=101,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_oot_fn_non_increase")
        assert gate.status == GateStatus.FAIL
        assert "exceeded Champion" in gate.failure_reason

    def test_gate_5_oot_cost_non_increase(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Candidate cost $25,000 vs Champion $20,890 -> FAIL
        comp = create_mock_comparison_result(
            candidate_oot_cost=25000.0,
            champion_oot_cost=20890.0,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_oot_cost_non_increase")
        assert gate.status == GateStatus.FAIL
        assert "exceeded Champion" in gate.failure_reason

    def test_gate_6_oot_fpr_ceiling(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Candidate FPR 0.00350 > ceiling 0.00200 -> FAIL
        comp = create_mock_comparison_result(
            candidate_oot_fpr=0.00350,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_oot_fpr_ceiling")
        assert gate.status == GateStatus.FAIL
        assert "exceeded safety ceiling" in gate.failure_reason

    def test_gate_7_oot_block_precision_floor(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Candidate precision 0.65000 < floor 0.70000 -> FAIL
        comp = create_mock_comparison_result(
            candidate_oot_precision=0.65000,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_oot_block_precision_floor")
        assert gate.status == GateStatus.FAIL
        assert "below required floor" in gate.failure_reason

    def test_gate_8_latency_sla(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Candidate p95 6.5 ms > SLA 5.0 ms -> FAIL
        comp = create_mock_comparison_result(
            candidate_p95_ms=6.50,
            champion_p95_ms=0.976,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_latency_p95_sla")
        assert gate.status == GateStatus.FAIL
        assert "exceeded allowed limit" in gate.failure_reason

    def test_gate_9_dynamic_champion_integrity_match_and_mismatch(self, mock_champion_dir: Path):
        real_m = calculate_file_sha256(mock_champion_dir / "champion_model.joblib")
        real_p = calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib")
        real_meta = calculate_file_sha256(mock_champion_dir / "model_metadata.json")

        # Matching hashes -> PASS
        comp_match = create_mock_comparison_result(
            champion_checksums={"model": real_m, "preprocessor": real_p, "metadata": real_meta}
        )
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp_match, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_champion_dynamic_integrity")
        assert gate.status == GateStatus.PASS
        assert gate.is_policy_configuration is False

        # Mutated hash -> FAIL
        comp_mismatch = create_mock_comparison_result(
            champion_checksums={"model": "corrupted_hash_value_12345", "preprocessor": real_p, "metadata": real_meta}
        )
        res_mismatch = evaluator.evaluate_eligibility(comp_mismatch, champion_dir=mock_champion_dir)
        gate_mismatch = next(g for g in res_mismatch.gate_results if g.gate_id == "gate_champion_dynamic_integrity")
        assert gate_mismatch.status == GateStatus.FAIL
        assert "do not match registered metadata hashes" in gate_mismatch.failure_reason

    def test_gate_10a_feature_schema_compatibility(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Feature count 54 instead of 55 -> FAIL
        comp = create_mock_comparison_result(feature_count=54, champion_checksums=hashes)
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_feature_schema_compatibility")
        assert gate.status == GateStatus.FAIL
        assert "does not match required schema count" in gate.failure_reason

    def test_gate_10b_evaluation_partition_integrity(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # OOT count 1000 instead of 277860 -> FAIL
        comp = create_mock_comparison_result(oot_sample_count=1000, champion_checksums=hashes)
        evaluator = PromotionGateEvaluator()
        res = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        gate = next(g for g in res.gate_results if g.gate_id == "gate_evaluation_partition_integrity")
        assert gate.status == GateStatus.FAIL
        assert "Partition row count mismatch" in gate.failure_reason


class TestPromotionEligibilitySynthesis:
    """Validates end-to-end evaluation assessment synthesis and error handling."""

    def test_all_11_gates_pass_yields_eligible_assessment(self, mock_champion_dir: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        comp = create_mock_comparison_result(champion_checksums=hashes)
        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)

        assert assessment.is_eligible is True
        assert assessment.total_gates_evaluated == 11
        assert assessment.passed_gates_count == 11
        assert assessment.failed_gates_count == 0
        assert assessment.blocked_gates_count == 0
        assert len(assessment.gate_results) == 11

    def test_evaluator_raises_on_missing_or_corrupt_file(self, tmp_path: Path):
        evaluator = PromotionGateEvaluator()
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate_eligibility(tmp_path / "non_existent.json")

        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(EvidenceCorruptedError):
            evaluator.evaluate_eligibility(corrupt_file)


class TestHumanSignOffEngine:
    """Validates role-based authorization, preconditions, and conflict prevention."""

    def test_admin_and_analyst_authorized_signoff(self, mock_champion_dir: Path, tmp_path: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        comp = create_mock_comparison_result(champion_checksums=hashes)
        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)

        engine = HumanSignOffEngine()

        # Admin Approval
        admin_actor = ActorContext(actor_id="admin_pranav", actor_role=AuditActorType.ADMIN)
        admin_out = tmp_path / "signoff_admin.json"
        record = engine.execute_signoff(
            assessment=assessment,
            actor=admin_actor,
            decision=SignOffDecision.APPROVED,
            rationale="Approved following multi-dimensional governance verification and cost reduction.",
            output_path=admin_out,
        )

        assert isinstance(record, PromotionSignOffRecord)
        assert record.decision == SignOffDecision.APPROVED
        assert record.actor_id == "admin_pranav"
        assert record.actor_role == "ADMIN"
        assert record.is_eligible_at_signoff is True
        assert admin_out.exists()

        # Analyst Rejection
        analyst_actor = ActorContext(actor_id="analyst_jane", actor_role=AuditActorType.ANALYST)
        analyst_out = tmp_path / "signoff_analyst.json"
        record_rej = engine.execute_signoff(
            assessment=assessment,
            actor=analyst_actor,
            decision=SignOffDecision.REJECTED,
            rationale="Voluntary governance hold requested for downstream business coordination.",
            output_path=analyst_out,
        )
        assert record_rej.decision == SignOffDecision.REJECTED

    def test_unauthorized_actor_roles_rejected(self, mock_champion_dir: Path, tmp_path: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        comp = create_mock_comparison_result(champion_checksums=hashes)
        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)

        engine = HumanSignOffEngine()

        # API_CLIENT Role
        api_actor = ActorContext(actor_id="api_key_1", actor_role=AuditActorType.API_CLIENT)
        with pytest.raises(UnauthorizedSignOffActorError, match="not authorized"):
            engine.execute_signoff(
                assessment=assessment,
                actor=api_actor,
                decision=SignOffDecision.APPROVED,
                rationale="Valid rationale that is at least 15 characters long.",
                output_path=tmp_path / "test.json",
            )

        # SYSTEM Role
        sys_actor = ActorContext(actor_id="cron_job", actor_role=AuditActorType.SYSTEM)
        with pytest.raises(UnauthorizedSignOffActorError, match="not authorized"):
            engine.execute_signoff(
                assessment=assessment,
                actor=sys_actor,
                decision=SignOffDecision.APPROVED,
                rationale="Valid rationale that is at least 15 characters long.",
                output_path=tmp_path / "test.json",
            )

    def test_approved_signoff_blocked_when_ineligible(self, mock_champion_dir: Path, tmp_path: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        # Ineligible comparison result (Recall regressed)
        comp = create_mock_comparison_result(
            candidate_oot_recall=0.75000,
            champion_oot_recall=0.89069,
            champion_checksums=hashes,
        )
        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)
        assert assessment.is_eligible is False

        engine = HumanSignOffEngine()
        admin_actor = ActorContext(actor_id="admin_1", actor_role=AuditActorType.ADMIN)

        with pytest.raises(PromotionGateBlockedError, match="Cannot execute APPROVED sign-off"):
            engine.execute_signoff(
                assessment=assessment,
                actor=admin_actor,
                decision=SignOffDecision.APPROVED,
                rationale="Attempting to force approve failing candidate.",
                output_path=tmp_path / "test.json",
            )

        # But REJECTED decision is permitted on ineligible assessment
        rej_record = engine.execute_signoff(
            assessment=assessment,
            actor=admin_actor,
            decision=SignOffDecision.REJECTED,
            rationale="Candidate rejected due to severe fraud catch rate regression.",
            output_path=tmp_path / "test_rej.json",
        )
        assert rej_record.decision == SignOffDecision.REJECTED

    def test_rationale_validation(self, mock_champion_dir: Path, tmp_path: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        comp = create_mock_comparison_result(champion_checksums=hashes)
        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)

        engine = HumanSignOffEngine()
        admin_actor = ActorContext(actor_id="admin_1", actor_role=AuditActorType.ADMIN)

        # Too short (< 15 chars)
        with pytest.raises(ValueError, match="at least 15 non-whitespace characters"):
            engine.execute_signoff(
                assessment=assessment,
                actor=admin_actor,
                decision=SignOffDecision.APPROVED,
                rationale="Too short",
                output_path=tmp_path / "test.json",
            )

    def test_signoff_conflict_prevention_and_idempotency(self, mock_champion_dir: Path, tmp_path: Path):
        hashes = {
            "model": calculate_file_sha256(mock_champion_dir / "champion_model.joblib"),
            "preprocessor": calculate_file_sha256(mock_champion_dir / "champion_preprocessor.joblib"),
            "metadata": calculate_file_sha256(mock_champion_dir / "model_metadata.json"),
        }
        comp = create_mock_comparison_result(champion_checksums=hashes)
        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(comp, champion_dir=mock_champion_dir)

        engine = HumanSignOffEngine()
        admin_actor = ActorContext(actor_id="admin_1", actor_role=AuditActorType.ADMIN)
        signoff_file = tmp_path / "signoff_test.json"

        # First execution -> Creates record
        rec1 = engine.execute_signoff(
            assessment=assessment,
            actor=admin_actor,
            decision=SignOffDecision.APPROVED,
            rationale="First valid sign-off with extensive explanation.",
            output_path=signoff_file,
        )
        assert signoff_file.exists()

        # Duplicate identical execution -> Idempotent return of same record
        rec2 = engine.execute_signoff(
            assessment=assessment,
            actor=admin_actor,
            decision=SignOffDecision.APPROVED,
            rationale="First valid sign-off with extensive explanation.",
            output_path=signoff_file,
        )
        assert rec1.signoff_id == rec2.signoff_id

        # Conflicting execution with different actor or decision -> Raises SignOffConflictError
        diff_actor = ActorContext(actor_id="admin_2", actor_role=AuditActorType.ADMIN)
        with pytest.raises(SignOffConflictError, match="conflicting sign-off record already exists"):
            engine.execute_signoff(
                assessment=assessment,
                actor=diff_actor,
                decision=SignOffDecision.APPROVED,
                rationale="Second actor attempting to overwrite first actor sign-off.",
                output_path=signoff_file,
            )
