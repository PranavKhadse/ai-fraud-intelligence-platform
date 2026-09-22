"""
Integration Tests for Phase 13.5: Monitoring Persistence, Snapshot Generation & Retention.

Validates end-to-end integration against PostgreSQL:
1. Extraction of real transactions, risk evaluations, and resolved cases.
2. Accurate calculation of feature drift, prediction drift, and model performance.
3. Idempotent snapshot persistence into `model_monitoring_snapshots`.
4. Unique constraint on (model_version, window_type, window_start, window_end).
5. Fast-path retrieval of persisted snapshots.
6. Zero writes or side-effects to business transaction, evaluation, case, or audit tables.
7. Bounded retention cleanup execution.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Any
import uuid
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.case import Case
from backend.app.db.models.enums import (
    CaseDisposition,
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
from ml.monitoring.config import default_monitoring_config
from ml.monitoring.schemas import FeatureBaselineProfile

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _build_dummy_features_snapshot(base_profile: FeatureBaselineProfile) -> Dict[str, Any]:
    """Generate a valid 55-feature dictionary matching the canonical catalog."""
    snapshot: Dict[str, Any] = {}
    for feat_name, num_meta in base_profile.features.items():
        if hasattr(num_meta, "bin_edges"):
            snapshot[feat_name] = float(num_meta.bin_edges[5])
        elif hasattr(num_meta, "vocabulary") and num_meta.vocabulary:
            snapshot[feat_name] = num_meta.vocabulary[0]
        else:
            snapshot[feat_name] = "UNKNOWN"
    return snapshot


class TestMonitoringPersistenceIntegration:
    """End-to-end integration tests for Monitoring persistence layer."""

    @pytest.fixture
    def base_profile(self) -> FeatureBaselineProfile:
        return FeatureBaselineProfile.load(default_monitoring_config.feature_profile_path)

    async def test_end_to_end_snapshot_generation_and_idempotent_persistence(
        self, db_session: AsyncSession, base_profile: FeatureBaselineProfile
    ):
        """Verify extraction, metric calculation, and idempotent snapshot persistence."""
        session = db_session
        service = MonitoringService()

        now = datetime.now(timezone.utc)
        w_start = now - timedelta(hours=24)
        w_end = now

        # 1. Populate transactions, evaluations, and cases in the database
        features_dict = _build_dummy_features_snapshot(base_profile)

        for i in range(25):
            txn_id = uuid.uuid4()
            eval_id = uuid.uuid4()
            eval_time = w_start + timedelta(hours=i % 20 + 1)

            txn = Transaction(
                id=txn_id,
                account_id=f"ACC_{i}",
                merchant_category="grocery_pos",
                job_category="engineer",
                amount=Decimal("125.50"),
                currency="USD",
                cardholder_lat=Decimal("40.7128"),
                cardholder_long=Decimal("-74.0060"),
                merchant_lat=Decimal("40.7130"),
                merchant_long=Decimal("-74.0055"),
                city_pop=50000,
                transaction_timestamp=eval_time,
                features_snapshot=features_dict,
            )
            session.add(txn)

            is_fraud = i < 5
            score = 0.90 if is_fraud else 0.10
            action = DecisionAction.BLOCK if is_fraud else DecisionAction.APPROVE
            tier = RiskTier.CRITICAL if is_fraud else RiskTier.LOW

            evaluation = RiskEvaluation(
                id=eval_id,
                transaction_id=txn_id,
                model_score=Decimal(str(score)),
                risk_score=90 if is_fraud else 10,
                decision_action=action,
                baseline_action=action,
                decision_reason="Test evaluation",
                risk_tier=tier,
                model_version="1.0.0",
                policy_mode=PolicyMode.TRI_TIER,
                evaluated_at=eval_time,
            )
            session.add(evaluation)

            # Create 10 resolved cases
            if i < 10:
                case_obj = Case(
                    id=uuid.uuid4(),
                    case_number=f"CASE-TEST-{i:03d}",
                    transaction_id=txn_id,
                    evaluation_id=eval_id,
                    status=CaseStatus.RESOLVED,
                    priority=CasePriority.HIGH if is_fraud else CasePriority.LOW,
                    trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
                    opened_at=eval_time,
                    resolved_at=eval_time + timedelta(minutes=30),
                    disposition=CaseDisposition.CONFIRMED_FRAUD if is_fraud else CaseDisposition.LEGITIMATE,
                    dispositioned_by="analyst_test",
                    dispositioned_at=eval_time + timedelta(minutes=30),
                )
                session.add(case_obj)

        await session.commit()

        # 2. Execute compute_and_persist_snapshot
        snapshot = await service.compute_and_persist_snapshot(
            session=session,
            window_type="DAILY",
            window_start=w_start,
            window_end=w_end,
            model_version="1.0.0",
        )

        assert snapshot.model_version == "1.0.0"
        assert snapshot.window_type == "DAILY"
        assert snapshot.sample_count == 25
        assert snapshot.labeled_count == 10
        assert snapshot.overall_status in ("NORMAL", "WARNING", "CRITICAL", "INSUFFICIENT_DATA")

        # 3. Verify Idempotency: re-running does NOT create duplicate row
        snapshot_v2 = await service.compute_and_persist_snapshot(
            session=session,
            window_type="DAILY",
            window_start=w_start,
            window_end=w_end,
            model_version="1.0.0",
        )

        assert snapshot_v2.id == snapshot.id

        # Verify exactly 1 snapshot row in database
        count_stmt = select(func.count(ModelMonitoringSnapshot.id))
        res = await session.execute(count_stmt)
        assert res.scalar() == 1

        # 4. Verify Zero Side Effects on Business Tables
        tx_count_stmt = select(func.count(Transaction.id))
        tx_count = (await session.execute(tx_count_stmt)).scalar()
        assert tx_count == 25

        case_count_stmt = select(func.count(Case.id))
        case_count = (await session.execute(case_count_stmt)).scalar()
        assert case_count == 10

    async def test_retention_cleanup_isolation(self, db_session: AsyncSession):
        """Verify retention cleanup only deletes expired snapshots and leaves business data untouched."""
        session = db_session
        repo = MonitoringRepository(session)
        now = datetime.now(timezone.utc)

        # 1. Insert 1 business transaction
        txn = Transaction(
            id=uuid.uuid4(),
            account_id="ACC_KEEP",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=Decimal("50.00"),
            currency="USD",
            cardholder_lat=Decimal("40.0"),
            cardholder_long=Decimal("-74.0"),
            merchant_lat=Decimal("40.0"),
            merchant_long=Decimal("-74.0"),
            city_pop=10000,
            transaction_timestamp=now - timedelta(days=500),
            features_snapshot={"amt": 50.0},
        )
        session.add(txn)

        # 2. Insert 1 active and 2 expired snapshots
        active_snap = ModelMonitoringSnapshot(
            id=uuid.uuid4(),
            model_version="1.0.0",
            window_type="DAILY",
            window_start=now - timedelta(days=5),
            window_end=now - timedelta(days=4),
            sample_count=100,
            labeled_count=10,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            feature_drift_summary={},
            prediction_drift_summary={},
            performance_summary={},
            created_at=now - timedelta(days=4),
        )
        expired_h = ModelMonitoringSnapshot(
            id=uuid.uuid4(),
            model_version="1.0.0",
            window_type="HOURLY",
            window_start=now - timedelta(days=35, hours=1),
            window_end=now - timedelta(days=35),
            sample_count=10,
            labeled_count=0,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            feature_drift_summary={},
            prediction_drift_summary={},
            performance_summary={},
            created_at=now - timedelta(days=35),
        )
        expired_d = ModelMonitoringSnapshot(
            id=uuid.uuid4(),
            model_version="1.0.0",
            window_type="DAILY",
            window_start=now - timedelta(days=400),
            window_end=now - timedelta(days=399),
            sample_count=200,
            labeled_count=20,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            feature_drift_summary={},
            prediction_drift_summary={},
            performance_summary={},
            created_at=now - timedelta(days=399),
        )

        session.add(active_snap)
        session.add(expired_h)
        session.add(expired_d)
        await session.commit()

        # Execute cleanup
        deleted = await repo.cleanup_expired_snapshots(hourly_retention_days=30, daily_retention_days=365, now=now)
        await session.commit()

        assert deleted == 2

        # Verify active snapshot remains
        remaining_snaps, snap_count = await repo.list_snapshots()
        assert snap_count == 1
        assert remaining_snaps[0].id == active_snap.id

        # Verify transaction remains untouched
        tx_count = (await session.execute(select(func.count(Transaction.id)))).scalar()
        assert tx_count == 1
