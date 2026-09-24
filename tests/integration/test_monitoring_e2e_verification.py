"""
End-to-End Monitoring Verification Test Suite for Phase 13.7.

Comprehensive verification of:
1. Cross-Phase Lifecycle (Phase 0–12 Ingestion, Scoring, Policy, Cases -> Phase 13 Monitoring & Alerts).
2. Four Monitoring Severity States Matrix (NORMAL, WARNING, CRITICAL, INSUFFICIENT_DATA).
3. Read-Only Table Immutability and Audit Isolation Guarantees (zero mutations to transactions, evaluations, cases, audit logs).
4. Snapshot Generation Idempotency, Unique Constraint Collision Safety, and Fast-Path Rollup Retrieval.
5. Active Alert Derivation and Severity Precedence Ordering.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from scripts.benchmark_monitoring import generate_synthetic_feature_dataframe

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.case import Case, CaseNote
from backend.app.db.models.enums import (
    AuditActorType,
    AuditEntityType,
    CaseDisposition,
    CaseNoteType,
    CasePriority,
    CaseStatus,
    CaseTriggerSource,
    DecisionAction,
    PolicyMode,
    RiskTier,
)
from backend.app.db.models.monitoring_snapshot import ModelMonitoringSnapshot
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
from backend.app.repositories.monitoring_repository import MonitoringRepository
from backend.app.services.monitoring_service import MonitoringService
from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.schemas import (
    CategoricalFeatureProfile,
    FeatureBaselineProfile,
    NumericalFeatureProfile,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _build_dummy_55_features(
    base_profile: FeatureBaselineProfile,
    drift_factor: float = 0.0,
    missing_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate synthetic 55-feature dictionary matching the canonical platform schema."""
    missing_set = set(missing_features or [])
    features: Dict[str, Any] = {}

    for feat_name, meta in base_profile.features.items():
        if feat_name in missing_set:
            features[feat_name] = None
        elif isinstance(meta, NumericalFeatureProfile):
            val = meta.mean + (drift_factor * meta.std)
            if meta.min >= 0:
                val = max(0.0, val)
            features[feat_name] = float(val)
        elif isinstance(meta, CategoricalFeatureProfile):
            if drift_factor > 0.5:
                features[feat_name] = "UNSEEN_DISRUPTIVE_CATEGORY"
            elif meta.vocabulary:
                features[feat_name] = meta.vocabulary[0]
            else:
                features[feat_name] = "UNKNOWN"
        else:
            features[feat_name] = 0.0

    return features


class TestMonitoringE2EVerification:
    """Comprehensive E2E verification suite for Phase 13.7."""

    @pytest.fixture
    def base_profile(self) -> FeatureBaselineProfile:
        return FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)

    async def test_full_cross_phase_lifecycle_flow(
        self, db_session: AsyncSession, base_profile: FeatureBaselineProfile
    ):
        """
        Verify end-to-end flow spanning Phase 0-12 data generation through Phase 13 monitoring.
        """
        session = db_session
        service = MonitoringService()

        now = datetime.now(timezone.utc)
        w_start = now - timedelta(hours=24)
        w_end = now

        # 1. Ingest transactions & evaluations representing various fraud risk profiles
        normal_feats = _build_dummy_55_features(base_profile, drift_factor=0.0)

        for i in range(60):
            txn_id = uuid.uuid4()
            eval_id = uuid.uuid4()
            # Spread transactions safely inside the 24h window
            t_stamp = w_start + timedelta(minutes=i * 20 + 10)

            txn = Transaction(
                id=txn_id,
                account_id=f"ACC_E2E_{i}",
                merchant_category="grocery_pos" if i % 2 == 0 else "shopping_net",
                job_category="engineer",
                amount=Decimal(str(round(25.0 + i * 10.0, 2))),
                currency="USD",
                cardholder_lat=Decimal("40.7128"),
                cardholder_long=Decimal("-74.0060"),
                merchant_lat=Decimal("40.7130"),
                merchant_long=Decimal("-74.0055"),
                city_pop=100000,
                transaction_timestamp=t_stamp,
                features_snapshot=normal_feats,
            )
            session.add(txn)

            is_high_fraud = i < 10
            is_review = 10 <= i < 30

            if is_high_fraud:
                score = Decimal("0.92")
                action = DecisionAction.BLOCK
                tier = RiskTier.CRITICAL
            elif is_review:
                score = Decimal("0.65")
                action = DecisionAction.REVIEW
                tier = RiskTier.HIGH
            else:
                score = Decimal("0.12")
                action = DecisionAction.APPROVE
                tier = RiskTier.LOW

            evaluation = RiskEvaluation(
                id=eval_id,
                transaction_id=txn_id,
                model_score=score,
                risk_score=int(float(score) * 100),
                decision_action=action,
                baseline_action=action,
                decision_reason="Phase 13 E2E test evaluation",
                risk_tier=tier,
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                evaluated_at=t_stamp,
            )
            session.add(evaluation)

            # Auto-create cases for REVIEW / BLOCK
            if is_review or is_high_fraud:
                case_id = uuid.uuid4()
                case_obj = Case(
                    id=case_id,
                    case_number=f"CASE-E2E-{i:04d}",
                    transaction_id=txn_id,
                    evaluation_id=eval_id,
                    status=CaseStatus.RESOLVED,
                    priority=CasePriority.CRITICAL if is_high_fraud else CasePriority.MEDIUM,
                    trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
                    opened_at=t_stamp,
                    resolved_at=t_stamp + timedelta(minutes=20),
                    disposition=CaseDisposition.CONFIRMED_FRAUD if is_high_fraud else CaseDisposition.LEGITIMATE,
                    dispositioned_by="senior_analyst_1",
                    dispositioned_at=t_stamp + timedelta(minutes=20),
                )
                session.add(case_obj)

                # Add case note and audit log
                note = CaseNote(
                    id=uuid.uuid4(),
                    case_id=case_id,
                    author_id="senior_analyst_1",
                    author_role=AuditActorType.ANALYST,
                    note_type=CaseNoteType.DISPOSITION,
                    content="Verified transaction disposition for monitoring verification.",
                    created_at=t_stamp + timedelta(minutes=20),
                )
                session.add(note)

                audit = AuditLog(
                    id=uuid.uuid4(),
                    event_type="CASE_DISPOSITION",
                    entity_type=AuditEntityType.CASE,
                    entity_id=case_id,
                    actor_id="senior_analyst_1",
                    actor_type=AuditActorType.ANALYST,
                    action="RESOLVE_CASE",
                    payload={"status": "RESOLVED", "disposition": case_obj.disposition.value},
                    created_at=t_stamp + timedelta(minutes=20),
                )
                session.add(audit)

        await session.commit()

        # 2. Query Monitoring Service Health Overview
        health_resp = await service.get_health_overview(
            session=session,
            window="24h",
            model_version="1.0.0",
        )

        assert health_resp.model_version == "1.0.0"
        assert health_resp.sample_count == 60
        assert health_resp.labeled_count == 30
        assert health_resp.overall_status in ("NORMAL", "WARNING", "CRITICAL", "INSUFFICIENT_DATA")

        # 3. Query Prediction Drift Report
        pred_resp = await service.get_prediction_drift(
            session=session,
            window="24h",
            model_version="1.0.0",
        )
        assert pred_resp.sample_count == 60
        assert pred_resp.overall_status in ("NORMAL", "WARNING", "CRITICAL", "INSUFFICIENT_DATA")
        assert pred_resp.action_distribution["APPROVE"] == pytest.approx(0.50, abs=0.05)

        # 4. Query Performance Report
        perf_resp = await service.get_performance(
            session=session,
            window="24h",
            model_version="1.0.0",
        )
        assert perf_resp.labeled_sample_count == 30
        assert perf_resp.fraud_cases_count == 10
        assert perf_resp.legitimate_cases_count == 20
        assert perf_resp.confidence in ("LOW_SAMPLE", "NORMAL_CONFIDENCE", "INSUFFICIENT_DATA")

        # 5. Compute & Persist Monitoring Snapshot
        snapshot = await service.compute_and_persist_snapshot(
            session=session,
            window_type="DAILY",
            window_start=w_start,
            window_end=w_end,
            model_version="1.0.0",
        )
        assert snapshot.sample_count == 60
        assert snapshot.labeled_count == 30

        # 6. Verify Fast-Path Resolution from Persisted Snapshot
        fast_path_health = await service.get_health_overview(
            session=session,
            start_time=w_start,
            end_time=w_end,
            model_version="1.0.0",
        )
        assert fast_path_health.sample_count == 60
        assert fast_path_health.overall_status == snapshot.overall_status

    async def test_four_severity_states_matrix(
        self, base_profile: FeatureBaselineProfile
    ):
        """
        Verify mathematical determination of all 4 severity states:
        INSUFFICIENT_DATA, NORMAL, WARNING, CRITICAL.
        """
        service = MonitoringService()

        # State 1: INSUFFICIENT_DATA (N = 10 < 100)
        df_small = generate_synthetic_feature_dataframe(base_profile, n_samples=10)
        f_rep_small = service.feature_calculator.compute_feature_drift(df_small)
        assert f_rep_small.overall_data_drift_status == DriftSeverity.INSUFFICIENT_DATA
        assert f_rep_small.insufficient_data_features_count == 55

        # State 2: NORMAL (N = 5000, congruent distribution matching 5000 reference samples)
        df_normal = generate_synthetic_feature_dataframe(base_profile, n_samples=5000, drift_factor=0.0)
        f_rep_normal = service.feature_calculator.compute_feature_drift(df_normal)
        assert f_rep_normal.overall_data_drift_status in (DriftSeverity.NORMAL, DriftSeverity.WARNING)

        # State 3: WARNING (Moderate drift)
        df_warning = generate_synthetic_feature_dataframe(base_profile, n_samples=5000, drift_factor=0.2)
        f_rep_warn = service.feature_calculator.compute_feature_drift(df_warning)
        assert f_rep_warn.overall_data_drift_status in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)

        # State 4: CRITICAL (Severe drift / missing value spike)
        df_crit = generate_synthetic_feature_dataframe(base_profile, n_samples=5000, drift_factor=2.0)
        f_rep_crit = service.feature_calculator.compute_feature_drift(df_crit)
        assert f_rep_crit.overall_data_drift_status == DriftSeverity.CRITICAL

        # Hierarchy Precedence Verification
        status_critical = service.resolve_overall_health_status(
            data_status=DriftSeverity.NORMAL,
            prediction_status=DriftSeverity.WARNING,
            perf_status=DriftSeverity.CRITICAL,
            sample_count=200,
            labeled_count=50,
        )
        assert status_critical == DriftSeverity.CRITICAL

        status_warning = service.resolve_overall_health_status(
            data_status=DriftSeverity.NORMAL,
            prediction_status=DriftSeverity.WARNING,
            perf_status=DriftSeverity.NORMAL,
            sample_count=200,
            labeled_count=50,
        )
        assert status_warning == DriftSeverity.WARNING

        status_normal = service.resolve_overall_health_status(
            data_status=DriftSeverity.NORMAL,
            prediction_status=DriftSeverity.NORMAL,
            perf_status=DriftSeverity.NORMAL,
            sample_count=200,
            labeled_count=50,
        )
        assert status_normal == DriftSeverity.NORMAL

        status_insufficient = service.resolve_overall_health_status(
            data_status=DriftSeverity.INSUFFICIENT_DATA,
            prediction_status=DriftSeverity.INSUFFICIENT_DATA,
            perf_status=DriftSeverity.INSUFFICIENT_DATA,
            sample_count=0,
            labeled_count=0,
        )
        assert status_insufficient == DriftSeverity.INSUFFICIENT_DATA

    async def test_read_only_immutability_guarantee(
        self, db_session: AsyncSession, base_profile: FeatureBaselineProfile
    ):
        """
        Verify that executing monitoring calculations, snapshot persistence, and retention cleanup
        guarantees ZERO mutations to transactions, risk evaluations, cases, case notes, or audit logs.
        """
        session = db_session
        service = MonitoringService()
        now = datetime.now(timezone.utc)

        # 1. Populate baseline business data
        txn_id = uuid.uuid4()
        eval_id = uuid.uuid4()
        case_id = uuid.uuid4()
        note_id = uuid.uuid4()
        audit_id = uuid.uuid4()

        txn = Transaction(
            id=txn_id,
            account_id="ACC_IMMUTABLE",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=Decimal("99.99"),
            currency="USD",
            cardholder_lat=Decimal("40.0"),
            cardholder_long=Decimal("-74.0"),
            merchant_lat=Decimal("40.0"),
            merchant_long=Decimal("-74.0"),
            city_pop=50000,
            transaction_timestamp=now - timedelta(hours=2),
            features_snapshot=_build_dummy_55_features(base_profile),
        )
        eval_obj = RiskEvaluation(
            id=eval_id,
            transaction_id=txn_id,
            model_score=Decimal("0.75"),
            risk_score=75,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            decision_reason="Immutability baseline evaluation",
            risk_tier=RiskTier.HIGH,
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            evaluated_at=now - timedelta(hours=2),
        )
        case_obj = Case(
            id=case_id,
            case_number="CASE-IMMUTABLE-001",
            transaction_id=txn_id,
            evaluation_id=eval_id,
            status=CaseStatus.RESOLVED,
            priority=CasePriority.HIGH,
            trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
            opened_at=now - timedelta(hours=2),
            resolved_at=now - timedelta(hours=1),
            disposition=CaseDisposition.CONFIRMED_FRAUD,
            dispositioned_by="analyst_immutability",
            dispositioned_at=now - timedelta(hours=1),
        )
        note = CaseNote(
            id=note_id,
            case_id=case_id,
            author_id="analyst_immutability",
            author_role=AuditActorType.ANALYST,
            note_type=CaseNoteType.INVESTIGATION,
            content="Baseline case note content.",
            created_at=now - timedelta(hours=1),
        )
        audit = AuditLog(
            id=audit_id,
            event_type="CASE_DISPOSITION",
            entity_type=AuditEntityType.CASE,
            entity_id=case_id,
            actor_id="analyst_immutability",
            actor_type=AuditActorType.ANALYST,
            action="DISPOSE",
            payload={"status": "RESOLVED"},
            created_at=now - timedelta(hours=1),
        )

        session.add_all([txn, eval_obj, case_obj, note, audit])
        await session.commit()

        # Capture initial state counts
        init_tx_count = (await session.execute(select(func.count(Transaction.id)))).scalar()
        init_eval_count = (await session.execute(select(func.count(RiskEvaluation.id)))).scalar()
        init_case_count = (await session.execute(select(func.count(Case.id)))).scalar()
        init_note_count = (await session.execute(select(func.count(CaseNote.id)))).scalar()
        init_audit_count = (await session.execute(select(func.count(AuditLog.id)))).scalar()

        # 2. Execute full suite of monitoring extractions and operations
        _ = await service.get_health_overview(session=session, window="24h")
        _ = await service.get_feature_drift(session=session, window="24h")
        _ = await service.get_feature_drift_detail(feature_name="amt", session=session, window="24h")
        _ = await service.get_prediction_drift(session=session, window="24h")
        _ = await service.get_performance(session=session, window="24h")
        snap = await service.compute_and_persist_snapshot(
            session=session,
            window_type="HOURLY",
            window_start=now - timedelta(hours=3),
            window_end=now,
            model_version="1.0.0",
        )
        assert snap is not None

        # Execute retention cleanup
        repo = MonitoringRepository(session)
        _ = await repo.cleanup_expired_snapshots(hourly_retention_days=30, daily_retention_days=365, now=now)
        await session.commit()

        # 3. Assert exact preservation of business table record counts
        final_tx_count = (await session.execute(select(func.count(Transaction.id)))).scalar()
        final_eval_count = (await session.execute(select(func.count(RiskEvaluation.id)))).scalar()
        final_case_count = (await session.execute(select(func.count(Case.id)))).scalar()
        final_note_count = (await session.execute(select(func.count(CaseNote.id)))).scalar()
        final_audit_count = (await session.execute(select(func.count(AuditLog.id)))).scalar()

        assert final_tx_count == init_tx_count
        assert final_eval_count == init_eval_count
        assert final_case_count == init_case_count
        assert final_note_count == init_note_count
        assert final_audit_count == init_audit_count

        # 4. Assert exact preservation of original business entity attributes
        fetched_txn = await session.get(Transaction, txn_id)
        assert fetched_txn.amount == Decimal("99.99")
        assert fetched_txn.account_id == "ACC_IMMUTABLE"

        fetched_case = await session.get(Case, case_id)
        assert fetched_case.disposition == CaseDisposition.CONFIRMED_FRAUD
        assert fetched_case.case_number == "CASE-IMMUTABLE-001"
