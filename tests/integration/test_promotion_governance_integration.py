"""
Integration Tests for Phase 14.5 Promotion Governance & Human Sign-Off.

Validates end-to-end:
1. Real Phase 14.4 comparison artifact consumption and cryptographic verification.
2. Complete 11-dimensional gate evaluation against real Champion and Candidate evidence.
3. Verification that Candidate v1.1.0 satisfies 100% of governance criteria (is_eligible = True).
4. Dynamic calculation of on-disk Champion SHA-256 hashes against registered metadata.
5. Authorized human sign-off execution in an isolated test fixture (zero fake production file side-effects).
6. PostgreSQL model registry state preservation (Candidate remains CANDIDATE, Champion remains active).
7. Disk manifest immutability (promotion_record remains null).
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import ActorContext
from backend.app.db.models.enums import AuditActorType
from backend.app.db.models.model_registry import ModelRegistryEntry
from backend.app.repositories.model_registry_repository import ModelRegistryRepository
from ml.lifecycle.config import default_lifecycle_config
from ml.lifecycle.governance import (
    GateStatus,
    HumanSignOffEngine,
    PromotionEligibilityAssessment,
    PromotionGateEvaluator,
    PromotionSignOffRecord,
    SignOffDecision,
    default_promotion_gate_specification,
)
from ml.lifecycle.schemas import ModelBundleManifest, ModelLifecycleStatus, calculate_file_sha256

COMPARISON_ARTIFACT_PATH = Path("ml/models/registry/comparisons/comparison_v1.0.0_vs_v1.1.0.json")
CHAMPION_DIR = Path("ml/models/artifacts")
CANDIDATE_BUNDLE_DIR = Path("ml/models/registry/candidates/v1.1.0")

pytestmark = pytest.mark.integration


class TestPromotionGovernanceIntegration:
    """Integration test suite for Phase 14.5 Governance Evaluation and Sign-Off."""

    def test_real_comparison_artifact_governance_evaluation(self):
        """
        Verify that PromotionGateEvaluator loads the real Phase 14.4 comparison artifact,
        dynamically hashes Champion artifacts, and evaluates all 11 gates.
        """
        assert COMPARISON_ARTIFACT_PATH.exists(), f"Phase 14.4 artifact missing at: {COMPARISON_ARTIFACT_PATH}"
        assert CHAMPION_DIR.exists(), f"Champion directory missing at: {CHAMPION_DIR}"

        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(
            comparison_evidence=COMPARISON_ARTIFACT_PATH,
            champion_dir=CHAMPION_DIR,
        )

        assert isinstance(assessment, PromotionEligibilityAssessment)
        assert assessment.candidate_version == "1.1.0"
        assert assessment.champion_version == "1.0.0"
        assert assessment.total_gates_evaluated == 11
        assert assessment.passed_gates_count == 11
        assert assessment.failed_gates_count == 0
        assert assessment.blocked_gates_count == 0
        assert assessment.is_eligible is True

        # Check evidence SHA-256 match
        expected_sha = calculate_file_sha256(COMPARISON_ARTIFACT_PATH)
        assert assessment.evidence_artifact_sha256 == expected_sha

        # Verify individual gate results on real evidence
        gates_by_id = {g.gate_id: g for g in assessment.gate_results}

        # Gate 1: OOT PR-AUC Parity
        g1 = gates_by_id["gate_oot_pr_auc_parity"]
        assert g1.status == GateStatus.PASS
        assert g1.actual_value == 0.96054
        assert g1.champion_value == 0.96054
        assert g1.delta_value == 0.0

        # Gate 2: Validation PR-AUC Parity
        g2 = gates_by_id["gate_val_pr_auc_parity"]
        assert g2.status == GateStatus.PASS
        assert g2.actual_value == 0.96190
        assert g2.champion_value == 0.96190

        # Gate 3: OOT Recall Non-Regression
        g3 = gates_by_id["gate_oot_recall_non_regression"]
        assert g3.status == GateStatus.PASS
        assert g3.actual_value == 0.94264
        assert g3.champion_value == 0.89069
        assert g3.delta_value == pytest.approx(0.05195, abs=1e-4)

        # Gate 4: OOT False Negatives Non-Increase
        g4 = gates_by_id["gate_oot_fn_non_increase"]
        assert g4.status == GateStatus.PASS
        assert g4.actual_value == 53
        assert g4.champion_value == 101
        assert g4.delta_value == -48

        # Gate 5: OOT Expected Cost Non-Increase
        g5 = gates_by_id["gate_oot_cost_non_increase"]
        assert g5.status == GateStatus.PASS
        assert g5.actual_value == 14245.0
        assert g5.champion_value == 20890.0
        assert g5.delta_value == -6645.0

        # Gate 6: OOT FPR Ceiling
        g6 = gates_by_id["gate_oot_fpr_ceiling"]
        assert g6.status == GateStatus.PASS
        assert g6.actual_value == 0.00088
        assert g6.actual_value <= 0.0020

        # Gate 7: OOT Block Precision Floor
        g7 = gates_by_id["gate_oot_block_precision_floor"]
        assert g7.status == GateStatus.PASS
        assert g7.actual_value == 0.78187
        assert g7.actual_value >= 0.7000

        # Gate 8: Inference Latency SLA
        g8 = gates_by_id["gate_latency_p95_sla"]
        assert g8.status == GateStatus.PASS
        assert g8.actual_value <= 5.0

        # Gate 9: Dynamic Champion Integrity
        g9 = gates_by_id["gate_champion_dynamic_integrity"]
        assert g9.status == GateStatus.PASS
        assert g9.actual_value == "hashes_match"

        # Gate 10A: Feature Schema Compatibility
        g10a = gates_by_id["gate_feature_schema_compatibility"]
        assert g10a.status == GateStatus.PASS
        assert g10a.actual_value == 55

        # Gate 10B: Partition Integrity
        g10b = gates_by_id["gate_evaluation_partition_integrity"]
        assert g10b.status == GateStatus.PASS
        assert g10b.actual_value == "val=277859, oot=277860"

    def test_isolated_human_signoff_execution_and_roundtrip(self, tmp_path: Path):
        """
        Verify that HumanSignOffEngine executes authorized sign-off into an isolated test fixture,
        without creating fake production sign-off artifacts.
        """
        evaluator = PromotionGateEvaluator()
        assessment = evaluator.evaluate_eligibility(
            comparison_evidence=COMPARISON_ARTIFACT_PATH,
            champion_dir=CHAMPION_DIR,
        )
        assert assessment.is_eligible is True

        engine = HumanSignOffEngine()
        test_signoff_file = tmp_path / "test_signoff_v1.1.0.json"

        admin_actor = ActorContext(
            actor_id="lead_risk_officer_01",
            actor_role=AuditActorType.ADMIN,
        )

        record = engine.execute_signoff(
            assessment=assessment,
            actor=admin_actor,
            decision=SignOffDecision.APPROVED,
            rationale="Candidate v1.1.0 approved for promotion: meets all 11 multi-dimensional criteria with 31.8% financial cost reduction.",
            output_path=test_signoff_file,
        )

        assert isinstance(record, PromotionSignOffRecord)
        assert record.candidate_version == "1.1.0"
        assert record.champion_version == "1.0.0"
        assert record.actor_id == "lead_risk_officer_01"
        assert record.actor_role == "ADMIN"
        assert record.decision == SignOffDecision.APPROVED
        assert record.is_eligible_at_signoff is True
        assert test_signoff_file.exists()

        # Verify round-trip JSON serialization
        with open(test_signoff_file, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        loaded_record = PromotionSignOffRecord.model_validate(loaded_data)
        assert loaded_record.signoff_id == record.signoff_id
        assert loaded_record.eligibility_assessment_sha256 == record.eligibility_assessment_sha256

        # Critical: Verify production sign-offs directory was NOT contaminated with test data
        prod_signoffs_dir = Path("ml/models/registry/signoffs")
        if prod_signoffs_dir.exists():
            assert not (prod_signoffs_dir / "test_signoff_v1.1.0.json").exists()

    @pytest.mark.asyncio
    async def test_database_and_disk_manifest_state_invariants(self, db_session: AsyncSession):
        """
        Verify that Phase 14.5 preserves all registry and manifest state invariants:
        - Candidate remains status = CANDIDATE
        - Candidate is_active_champion remains False
        - Active champion remains 1.0.0
        - Manifest promotion_record remains null
        """
        repo = ModelRegistryRepository(db_session)
        # Ensure champion and candidate are present in test db session
        champ_entry, _ = await repo.register_initial_champion_if_empty()
        cand_entry = await repo.get_by_version("1.1.0")
        if cand_entry is None:
            cand_manifest_p = CANDIDATE_BUNDLE_DIR / "manifest.json"
            cand_manifest = ModelBundleManifest.load(cand_manifest_p)
            cand_entry = ModelRegistryEntry(
                id=uuid.uuid4(),
                model_version="1.1.0",
                model_family="xgboost",
                status=cand_manifest.status.value,
                is_active_champion=False,
                operating_threshold=Decimal(str(cand_manifest.operating_threshold)),
                bundle_directory=str(CANDIDATE_BUNDLE_DIR),
                sha256_model=cand_manifest.sha256_checksums["model"],
                sha256_preprocessor=cand_manifest.sha256_checksums["preprocessor"],
                validation_metrics=cand_manifest.validation_metrics if isinstance(cand_manifest.validation_metrics, dict) else (cand_manifest.validation_metrics.model_dump() if cand_manifest.validation_metrics else None),
                oot_metrics=cand_manifest.oot_holdout_metrics if isinstance(cand_manifest.oot_holdout_metrics, dict) else (cand_manifest.oot_holdout_metrics.model_dump() if cand_manifest.oot_holdout_metrics else None),
            )
            await repo.create_entry(cand_entry)
            await db_session.commit()

        # 1. PostgreSQL Candidate Entry
        cand_entry = await repo.get_by_version("1.1.0")
        assert cand_entry is not None, "Candidate v1.1.0 missing from model_registry_entries"
        assert cand_entry.status == ModelLifecycleStatus.CANDIDATE.value
        assert cand_entry.is_active_champion is False
        assert cand_entry.promoted_at is None
        assert cand_entry.promoted_by is None
        assert cand_entry.promotion_rationale is None

        # 2. PostgreSQL Active Champion Entry
        champ_entry = await repo.get_active_champion()
        assert champ_entry is not None, "Active champion missing from database"
        assert champ_entry.model_version == "1.0.0"
        assert champ_entry.is_active_champion is True
        assert champ_entry.status == ModelLifecycleStatus.CHAMPION.value

        # 3. Disk Manifest Invariant
        manifest_p = CANDIDATE_BUNDLE_DIR / "manifest.json"
        assert manifest_p.exists()
        manifest = ModelBundleManifest.load(manifest_p)
        assert manifest.model_version == "1.1.0"
        assert manifest.status == ModelLifecycleStatus.CANDIDATE
        assert manifest.promotion_record is None

        # 4. Dynamic Champion Checksums on Disk
        m_sha = calculate_file_sha256(CHAMPION_DIR / "champion_model.joblib")
        p_sha = calculate_file_sha256(CHAMPION_DIR / "champion_preprocessor.joblib")
        meta_sha = calculate_file_sha256(CHAMPION_DIR / "model_metadata.json")

        assert m_sha.lower() == cand_entry.sha256_model.lower()  # In Phase 14.2/3 champion binary was used as initial weights
        assert p_sha.lower() == cand_entry.sha256_preprocessor.lower()
        assert meta_sha.lower() == "deb0f0e8e8b5436fb38c84e0d9554de1f6ce8097b3b57af70aea02abefc35c27"
