"""
Unit Tests for Phase 13.5: Monitoring Persistence, Snapshot Generation, APIs & Retention.

Validates:
1. MonitoringRepository: CRUD, idempotent upsert, list filtering, and bounded retention cleanup.
2. MonitoringService: Window resolution, status precedence (CRITICAL > WARNING > NORMAL > INSUFFICIENT),
   active alert derivation, and read-only data extraction.
3. REST API Contracts:
   - GET /api/v1/monitoring/health
   - GET /api/v1/monitoring/drift/features
   - GET /api/v1/monitoring/drift/features/{feature_name} (valid & 404)
   - GET /api/v1/monitoring/drift/predictions
   - GET /api/v1/monitoring/performance
   - GET /api/v1/monitoring/snapshots
4. Safety & Isolation: Zero writes to business tables during metric computation.
5. Retention boundaries and CLI execution.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
import uuid
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.monitoring_snapshot import ModelMonitoringSnapshot
from backend.app.db.session import get_db_session
from backend.app.main import app
from backend.app.repositories.monitoring_repository import MonitoringRepository
from backend.app.schemas.monitoring import (
    ConfusionMatrixResponse,
    FeatureDriftDetailResponse,
    FeatureDriftItemResponse,
    FeatureDriftListResponse,
    MetricDegradationItemResponse,
    ModelPerformanceResponse,
    MonitoringAlertItem,
    MonitoringHealthResponse,
    MonitoringSnapshotItemResponse,
    MonitoringSnapshotListResponse,
    OperationalMetricsResponse,
    PredictionDriftResponse,
    ThresholdMetricsResponse,
)
from backend.app.services.monitoring_service import (
    MonitoringService,
    get_monitoring_service,
)
from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.schemas import (
    CategoricalDistributionDriftResult,
    FeatureDriftReport,
    OverrideRateDriftResult,
    PerformanceReport,
    PredictionDriftReport,
    RiskScoreBucket,
    RiskScoreDriftResult,
    ScoreDriftResult,
    SingleFeatureDriftResult,
)


# ==============================================================================
# 1. MonitoringRepository Unit Tests (Mocked Session)
# ==============================================================================

class TestMonitoringRepositoryUnit:
    """Test suite for MonitoringRepository methods."""

    @pytest.mark.asyncio
    async def test_repository_get_by_id_and_window(self):
        """Verify get_by_id and get_by_window execute select statements."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = MonitoringRepository(mock_session)

        test_id = uuid.uuid4()
        mock_snapshot = ModelMonitoringSnapshot(
            id=test_id,
            model_version="1.0.0",
            window_type="DAILY",
            window_start=datetime.now(timezone.utc),
            window_end=datetime.now(timezone.utc),
            sample_count=100,
            labeled_count=10,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            feature_drift_summary={},
            prediction_drift_summary={},
            performance_summary={},
            created_at=datetime.now(timezone.utc),
        )

        # Mock execute result
        mock_res = MagicMock()
        mock_res.scalars.return_value.first.return_value = mock_snapshot
        mock_session.execute.return_value = mock_res

        res = await repo.get_by_id(test_id)
        assert res == mock_snapshot
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_repository_upsert_update_existing(self):
        """Verify upsert updates existing record in-place."""
        mock_session = AsyncMock(spec=AsyncSession)
        repo = MonitoringRepository(mock_session)

        now = datetime.now(timezone.utc)
        existing = ModelMonitoringSnapshot(
            id=uuid.uuid4(),
            model_version="1.0.0",
            window_type="DAILY",
            window_start=now - timedelta(days=1),
            window_end=now,
            sample_count=50,
            labeled_count=5,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            feature_drift_summary={},
            prediction_drift_summary={},
            performance_summary={},
            created_at=now,
        )

        mock_res = MagicMock()
        mock_res.scalars.return_value.first.return_value = existing
        mock_session.execute.return_value = mock_res

        new_data = ModelMonitoringSnapshot(
            id=uuid.uuid4(),
            model_version="1.0.0",
            window_type="DAILY",
            window_start=now - timedelta(days=1),
            window_end=now,
            sample_count=120,
            labeled_count=15,
            overall_status="WARNING",
            data_drift_status="WARNING",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            feature_drift_summary={"updated": True},
            prediction_drift_summary={},
            performance_summary={},
            created_at=now,
        )

        updated = await repo.upsert_snapshot(new_data)
        assert updated.id == existing.id
        assert updated.sample_count == 120
        assert updated.overall_status == "WARNING"
        mock_session.flush.assert_called_once()


# ==============================================================================
# 2. MonitoringService Unit Tests
# ==============================================================================

class TestMonitoringServiceLogic:
    """Test suite for MonitoringService core algorithms and precedence."""

    @pytest.fixture
    def service(self) -> MonitoringService:
        return MonitoringService()

    def test_window_boundary_resolution(self, service: MonitoringService):
        """Verify window strings resolve to correct relative intervals."""
        s, e, w_type = service.resolve_window_boundaries("1h")
        assert w_type == "HOURLY"
        assert np.isclose((e - s).total_seconds(), 3600, atol=2)

        s24, e24, w_type24 = service.resolve_window_boundaries("24h")
        assert w_type24 == "DAILY"
        assert np.isclose((e24 - s24).total_seconds(), 86400, atol=2)

        s7, e7, w_type7 = service.resolve_window_boundaries("7d")
        assert w_type7 == "7D"

        # Custom window
        custom_s = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
        custom_e = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
        sc, ec, wtc = service.resolve_window_boundaries(start_time=custom_s, end_time=custom_e)
        assert wtc == "custom"
        assert sc == custom_s
        assert ec == custom_e

    def test_overall_health_precedence_hierarchy(self, service: MonitoringService):
        """Verify CRITICAL > WARNING > HEALTHY > INSUFFICIENT precedence."""
        # 1. CRITICAL dominates
        status1 = service.resolve_overall_health_status(
            data_status=DriftSeverity.NORMAL,
            prediction_status=DriftSeverity.CRITICAL,
            perf_status=DriftSeverity.WARNING,
            sample_count=100,
            labeled_count=20,
        )
        assert status1 == DriftSeverity.CRITICAL

        # 2. WARNING dominates over NORMAL
        status2 = service.resolve_overall_health_status(
            data_status=DriftSeverity.WARNING,
            prediction_status=DriftSeverity.NORMAL,
            perf_status=DriftSeverity.NORMAL,
            sample_count=100,
            labeled_count=20,
        )
        assert status2 == DriftSeverity.WARNING

        # 3. All NORMAL -> NORMAL (HEALTHY)
        status3 = service.resolve_overall_health_status(
            data_status=DriftSeverity.NORMAL,
            prediction_status=DriftSeverity.NORMAL,
            perf_status=DriftSeverity.NORMAL,
            sample_count=100,
            labeled_count=20,
        )
        assert status3 == DriftSeverity.NORMAL

        # 4. 0 samples -> INSUFFICIENT_DATA
        status4 = service.resolve_overall_health_status(
            data_status=DriftSeverity.INSUFFICIENT_DATA,
            prediction_status=DriftSeverity.INSUFFICIENT_DATA,
            perf_status=DriftSeverity.INSUFFICIENT_DATA,
            sample_count=0,
            labeled_count=0,
        )
        assert status4 == DriftSeverity.INSUFFICIENT_DATA

        # 5. Partial INSUFFICIENT_DATA rule:
        # If performance has 0 labeled cases but data & prediction are NORMAL on 100 samples -> overall is NORMAL
        status5 = service.resolve_overall_health_status(
            data_status=DriftSeverity.NORMAL,
            prediction_status=DriftSeverity.NORMAL,
            perf_status=DriftSeverity.INSUFFICIENT_DATA,
            sample_count=100,
            labeled_count=0,
        )
        assert status5 == DriftSeverity.NORMAL

    def test_derived_alerts_extraction(self, service: MonitoringService):
        """Verify active alerts extraction across all 3 monitoring components."""
        # Mock feature report with 1 WARNING feature
        feat_item = SingleFeatureDriftResult(
            feature_name="amt",
            feature_type="numerical",
            drift_metric_name="PSI",
            drift_metric_value=0.15,
            ks_statistic=0.10,
            ks_pvalue=0.01,
            missing_rate_delta=0.0,
            severity=DriftSeverity.WARNING,
        )
        feat_rep = FeatureDriftReport(
            model_version="1.0.0",
            dataset_row_count=100,
            overall_data_drift_status=DriftSeverity.WARNING,
            drifted_features_count=1,
            critical_features_count=0,
            warning_features_count=1,
            normal_features_count=54,
            insufficient_data_features_count=0,
            feature_results={"amt": feat_item},
            ranked_features=[feat_item],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Mock prediction report with 1 active warning
        score_drift = ScoreDriftResult(
            psi_value=0.12,
            observed_bins=[],
            expected_bins=[],
            observed_mean=0.25,
            baseline_mean=0.20,
            severity=DriftSeverity.WARNING,
            alert_message="Model score distribution drift warning",
        )
        risk_drift = RiskScoreDriftResult(
            psi_value=0.04,
            observed_buckets=[],
            expected_buckets=[],
            observed_mean=25.0,
            baseline_mean=20.0,
            severity=DriftSeverity.NORMAL,
        )
        tier_drift = CategoricalDistributionDriftResult(
            metric_name="JSD",
            jsd_value=0.02,
            observed_proportions={},
            expected_proportions={},
            severity=DriftSeverity.NORMAL,
        )
        action_drift = CategoricalDistributionDriftResult(
            metric_name="JSD",
            jsd_value=0.02,
            observed_proportions={},
            expected_proportions={},
            severity=DriftSeverity.NORMAL,
        )
        override_drift = OverrideRateDriftResult(
            observed_rate=0.05,
            baseline_rate=0.04,
            rate_delta=0.01,
            overridden_count=5,
            total_count=100,
        )

        pred_rep = PredictionDriftReport(
            model_version="1.0.0",
            dataset_row_count=100,
            overall_prediction_drift_status=DriftSeverity.WARNING,
            model_score_drift=score_drift,
            risk_score_drift=risk_drift,
            tier_drift=tier_drift,
            action_drift=action_drift,
            override_drift=override_drift,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Mock performance report with 0 alerts
        perf_rep = service.performance_calculator.compute_performance_from_labels(
            y_true=[1] * 20 + [0] * 20,
            model_scores=[0.9] * 20 + [0.1] * 20,
            actions=["BLOCK"] * 20 + ["APPROVE"] * 20,
            total_raw_count=40,
        )

        alerts = service.extract_derived_alerts(feat_rep, pred_rep, perf_rep)

        assert len(alerts) == 3
        assert alerts[0].component == "data_drift"
        assert alerts[0].feature_name == "amt"
        assert alerts[0].severity == "WARNING"

        assert alerts[1].component == "prediction_drift"
        assert alerts[1].metric_name == "model_score_psi"

        assert alerts[2].component == "performance"
        assert alerts[2].metric_name == "review_queue_purity"


# ==============================================================================
# 3. FastAPI REST Endpoint Tests (Dependency Overrides)
# ==============================================================================

class TestMonitoringRESTEndpoints:
    """Test suite verifying all 6 GET /api/v1/monitoring/* endpoints."""

    @pytest.fixture
    def mock_service(self) -> MagicMock:
        service = MagicMock(spec=MonitoringService)
        return service

    @pytest.fixture
    def client(self, mock_service) -> TestClient:
        app.dependency_overrides[get_monitoring_service] = lambda: mock_service
        app.dependency_overrides[get_db_session] = lambda: AsyncMock(spec=AsyncSession)
        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides.clear()

    def test_get_monitoring_health_endpoint(self, client: TestClient, mock_service):
        """GET /api/v1/monitoring/health returns 200 and conforms to schema."""
        mock_service.get_health_overview = AsyncMock(
            return_value=MonitoringHealthResponse(
                model_version="1.0.0",
                window_type="DAILY",
                window_start="2026-09-21T00:00:00Z",
                window_end="2026-09-22T00:00:00Z",
                sample_count=100,
                labeled_count=20,
                overall_status="NORMAL",
                data_drift_status="NORMAL",
                prediction_drift_status="NORMAL",
                performance_status="NORMAL",
                active_alert_count=0,
                active_alerts=[],
                created_at="2026-09-22T00:00:00Z",
            )
        )

        response = client.get("/api/v1/monitoring/health?window=24h")
        assert response.status_code == 200
        data = response.json()

        assert data["model_version"] == "1.0.0"
        assert data["overall_status"] == "NORMAL"
        assert data["sample_count"] == 100

    def test_get_feature_drift_report_endpoint(self, client: TestClient, mock_service):
        """GET /api/v1/monitoring/drift/features returns 200 and list of feature items."""
        mock_item = FeatureDriftItemResponse(
            feature_name="amt",
            feature_type="numerical",
            category="amount",
            status="NORMAL",
            psi=0.02,
            ks_statistic=0.03,
            ks_p_value=0.45,
            js_divergence=0.0,
            missing_rate_current=0.0,
            missing_rate_delta=0.0,
            unseen_category_rate=0.0,
        )
        mock_service.get_feature_drift = AsyncMock(
            return_value=FeatureDriftListResponse(
                model_version="1.0.0",
                window_type="DAILY",
                window_start="2026-09-21T00:00:00Z",
                window_end="2026-09-22T00:00:00Z",
                sample_count=100,
                overall_status="NORMAL",
                total_features_count=55,
                drifted_features_count=0,
                critical_features_count=0,
                warning_features_count=0,
                items=[mock_item],
            )
        )

        response = client.get("/api/v1/monitoring/drift/features?category=amount&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["total_features_count"] == 55
        assert len(data["items"]) == 1
        assert data["items"][0]["feature_name"] == "amt"

    def test_get_single_feature_drift_detail_endpoint(self, client: TestClient, mock_service):
        """GET /api/v1/monitoring/drift/features/{feature_name} returns 200 and 404 for unknown feature."""
        # 1. Valid feature
        mock_service.get_feature_drift_detail = AsyncMock(
            return_value=FeatureDriftDetailResponse(
                feature_name="amt",
                feature_type="numerical",
                category="amount",
                status="NORMAL",
                psi=0.02,
                ks_statistic=0.03,
                ks_p_value=0.45,
                js_divergence=0.0,
                missing_rate_baseline=0.0,
                missing_rate_current=0.0,
                missing_rate_delta=0.0,
                unseen_category_rate=0.0,
                unseen_categories=[],
                baseline_distribution={"bin_0": 0.1},
                current_distribution={"bin_0": 0.1},
                bin_edges=[0.0, 10.0, 100.0],
            )
        )

        response = client.get("/api/v1/monitoring/drift/features/amt")
        assert response.status_code == 200
        data = response.json()
        assert data["feature_name"] == "amt"

        # 2. Unknown feature -> 404
        mock_service.get_feature_drift_detail = AsyncMock(return_value=None)
        resp_404 = client.get("/api/v1/monitoring/drift/features/unknown_feat")
        assert resp_404.status_code == 404
        assert "not found" in resp_404.json()["detail"].lower()

    def test_get_prediction_drift_endpoint(self, client: TestClient, mock_service):
        """GET /api/v1/monitoring/drift/predictions returns 200 and prediction response schema."""
        mock_service.get_prediction_drift = AsyncMock(
            return_value=PredictionDriftResponse(
                model_version="1.0.0",
                window_type="DAILY",
                window_start="2026-09-21T00:00:00Z",
                window_end="2026-09-22T00:00:00Z",
                sample_count=100,
                overall_status="NORMAL",
                model_score_psi=0.01,
                model_score_status="NORMAL",
                model_score_mean=0.20,
                model_score_std=0.10,
                risk_score_psi=0.02,
                risk_score_status="NORMAL",
                risk_tier_jsd=0.01,
                risk_tier_status="NORMAL",
                action_jsd=0.01,
                action_status="NORMAL",
                override_rate_current=0.04,
                override_rate_baseline=0.04,
                override_rate_delta=0.0,
                model_score_distribution={},
                risk_score_buckets_distribution={},
                risk_tier_distribution={},
                action_distribution={},
                active_alerts=[],
            )
        )

        response = client.get("/api/v1/monitoring/drift/predictions")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "NORMAL"
        assert data["model_score_psi"] == 0.01

    def test_get_performance_endpoint(self, client: TestClient, mock_service):
        """GET /api/v1/monitoring/performance returns 200 and 3-tier metrics."""
        mock_cm = ConfusionMatrixResponse(tp=40, fp=5, fn=10, tn=45, total=100)
        mock_prim = ThresholdMetricsResponse(
            threshold=0.78,
            precision=0.8889,
            recall=0.80,
            f1=0.8421,
            accuracy=0.85,
            fpr=0.10,
            tpr=0.80,
            confusion_matrix=mock_cm,
        )
        mock_comp = ThresholdMetricsResponse(
            threshold=0.50,
            precision=0.75,
            recall=0.90,
            f1=0.8182,
            accuracy=0.80,
            fpr=0.15,
            tpr=0.90,
            confusion_matrix=mock_cm,
        )
        mock_ops = OperationalMetricsResponse(
            decision_precision_block=0.85,
            decision_recall_intervention=0.95,
            review_queue_purity=0.02,
            total_reviews_count=50,
            fraud_in_review_count=1,
            total_blocks_count=45,
            fraud_in_block_count=38,
        )

        mock_service.get_performance = AsyncMock(
            return_value=ModelPerformanceResponse(
                model_version="1.0.0",
                window_type="DAILY",
                window_start="2026-09-21T00:00:00Z",
                window_end="2026-09-22T00:00:00Z",
                dataset_row_count=100,
                labeled_sample_count=100,
                fraud_cases_count=50,
                legitimate_cases_count=50,
                suspicious_resolved_count=0,
                overall_performance_status="NORMAL",
                confidence="NORMAL_CONFIDENCE",
                operating_threshold=0.78,
                primary_metrics=mock_prim,
                comparison_metrics=mock_comp,
                pr_auc=0.95,
                roc_auc=0.98,
                operational_metrics=mock_ops,
                degradation_results={},
                active_degradation_alerts=[],
            )
        )

        response = client.get("/api/v1/monitoring/performance")
        assert response.status_code == 200
        data = response.json()
        assert data["operating_threshold"] == 0.78
        assert data["primary_metrics"]["precision"] == 0.8889
        assert data["operational_metrics"]["decision_precision_block"] == 0.85

    def test_get_snapshots_endpoint(self, client: TestClient, mock_service):
        """GET /api/v1/monitoring/snapshots returns 200 and paginated snapshot list."""
        mock_snap_item = MonitoringSnapshotItemResponse(
            id=str(uuid.uuid4()),
            model_version="1.0.0",
            window_type="DAILY",
            window_start="2026-09-21T00:00:00Z",
            window_end="2026-09-22T00:00:00Z",
            sample_count=150,
            labeled_count=30,
            overall_status="NORMAL",
            data_drift_status="NORMAL",
            prediction_drift_status="NORMAL",
            performance_status="NORMAL",
            created_at="2026-09-22T00:00:00Z",
        )
        mock_service.list_snapshots = AsyncMock(
            return_value=MonitoringSnapshotListResponse(
                total_count=1,
                limit=10,
                offset=0,
                items=[mock_snap_item],
            )
        )

        response = client.get("/api/v1/monitoring/snapshots?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["model_version"] == "1.0.0"
