"""
Monitoring Service for Phase 13 ML & Model Monitoring.

Orchestrates:
- Feature drift, prediction drift, and ground-truth performance evaluation engines.
- Read-only data extraction from Transaction, RiskEvaluation, and Case records.
- Idempotent HOURLY and DAILY snapshot computation and persistence into `model_monitoring_snapshots`.
- Health status precedence resolution and derived alert generation.
- Service methods for REST API endpoints with on-demand computation and persisted snapshot fast-paths.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models.case import Case
from backend.app.db.models.monitoring_snapshot import ModelMonitoringSnapshot
from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.db.models.transaction import Transaction
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
from ml.monitoring.config import (
    DriftSeverity,
    MetricConfidence,
    MonitoringConfig,
    default_monitoring_config,
)
from ml.monitoring.feature_drift import FeatureDriftCalculator
from ml.monitoring.performance import ModelPerformanceCalculator
from ml.monitoring.prediction_drift import PredictionDriftCalculator
from ml.monitoring.schemas import (
    CategoricalFeatureProfile,
    FeatureDriftReport,
    NumericalFeatureProfile,
    PerformanceReport,
    PredictionDriftReport,
    SingleFeatureDriftResult,
)

logger = logging.getLogger("fraud_api.monitoring")


def _get_feature_category(feat_name: str) -> str:
    """Classify feature into functional category group."""
    feat_lower = feat_name.lower()
    if any(k in feat_lower for k in ("amt", "amount", "ratio")):
        return "amount"
    if any(k in feat_lower for k in ("cnt", "velocity", "freq", "window", "hour")):
        return "velocity"
    if any(k in feat_lower for k in ("dist", "lat", "long", "city", "geo")):
        return "geographic"
    if any(k in feat_lower for k in ("age", "gender", "job", "dob")):
        return "demographic"
    return "behavioral"


class MonitoringService:
    """
    Central orchestration service for Phase 13 Model Monitoring.
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        feature_calculator: Optional[FeatureDriftCalculator] = None,
        prediction_calculator: Optional[PredictionDriftCalculator] = None,
        performance_calculator: Optional[ModelPerformanceCalculator] = None,
    ) -> None:
        self.config = config or default_monitoring_config
        self.feature_calculator = feature_calculator or FeatureDriftCalculator(config=self.config)
        self.prediction_calculator = prediction_calculator or PredictionDriftCalculator(config=self.config)
        self.performance_calculator = performance_calculator or ModelPerformanceCalculator(config=self.config)

    def resolve_window_boundaries(
        self,
        window: Optional[str] = "24h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Tuple[datetime, datetime, str]:
        """
        Resolve start/end timestamps and window type from query parameters.
        """
        now = datetime.now(timezone.utc)

        if start_time is not None and end_time is not None:
            s_time = start_time if start_time.tzinfo else start_time.replace(tzinfo=timezone.utc)
            e_time = end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)
            return s_time, e_time, "custom"

        win = (window or "24h").lower().strip()
        if win == "1h":
            return now - timedelta(hours=1), now, "HOURLY"
        elif win in ("24h", "1d"):
            return now - timedelta(hours=24), now, "DAILY"
        elif win == "7d":
            return now - timedelta(days=7), now, "7D"
        elif win == "30d":
            return now - timedelta(days=30), now, "30D"
        else:
            return now - timedelta(hours=24), now, "DAILY"

    async def extract_monitoring_data(
        self,
        session: AsyncSession,
        start_time: datetime,
        end_time: datetime,
        model_version: str = "1.0.0",
    ) -> Tuple[pd.DataFrame, List[RiskEvaluation], List[Case]]:
        """
        Perform read-only query extraction of evaluated transactions and resolved cases in the window.

        Guarantees:
        - Zero writes or state changes to business tables.
        """
        # 1. Query RiskEvaluations and joined Transactions
        eval_stmt = (
            select(RiskEvaluation)
            .options(selectinload(RiskEvaluation.transaction))
            .where(
                RiskEvaluation.evaluated_at >= start_time,
                RiskEvaluation.evaluated_at < end_time,
                RiskEvaluation.model_version == model_version,
            )
            .order_by(RiskEvaluation.evaluated_at.asc())
        )
        eval_res = await session.execute(eval_stmt)
        evaluations: List[RiskEvaluation] = list(eval_res.scalars().all())

        # Build feature DataFrame from Transaction.features_snapshot
        feature_rows: List[Dict[str, Any]] = []
        for ev in evaluations:
            if ev.transaction and ev.transaction.features_snapshot:
                feature_rows.append(ev.transaction.features_snapshot)

        df_features = pd.DataFrame(feature_rows) if feature_rows else pd.DataFrame()

        # 2. Query resolved Cases with linked RiskEvaluation
        case_stmt = (
            select(Case)
            .options(selectinload(Case.evaluation))
            .where(
                Case.disposition.is_not(None),
                Case.evaluation_id.in_([ev.id for ev in evaluations]) if evaluations else Case.id.is_not(None),
            )
        )
        if not evaluations:
            case_stmt = (
                select(Case)
                .options(selectinload(Case.evaluation))
                .where(
                    Case.disposition.is_not(None),
                    Case.resolved_at >= start_time,
                    Case.resolved_at < end_time,
                )
            )

        case_res = await session.execute(case_stmt)
        cases: List[Case] = list(case_res.scalars().all())

        return df_features, evaluations, cases

    def resolve_overall_health_status(
        self,
        data_status: DriftSeverity,
        prediction_status: DriftSeverity,
        perf_status: DriftSeverity,
        sample_count: int,
        labeled_count: int,
    ) -> DriftSeverity:
        """
        Determine overall monitoring health status using strict precedence:
        1. CRITICAL if ANY component is CRITICAL.
        2. WARNING if ANY component is WARNING.
        3. HEALTHY (NORMAL) if evaluated components with sufficient data are NORMAL.
        4. INSUFFICIENT_DATA if all components have insufficient data.
        5. Partial rule: If perf_status is INSUFFICIENT_DATA but data/prediction are NORMAL, overall is NORMAL.
        """
        all_statuses = [data_status, prediction_status, perf_status]

        if any(s == DriftSeverity.CRITICAL for s in all_statuses):
            return DriftSeverity.CRITICAL

        if any(s == DriftSeverity.WARNING for s in all_statuses):
            return DriftSeverity.WARNING

        # Check if all are INSUFFICIENT_DATA or sample_count is 0
        if all(s == DriftSeverity.INSUFFICIENT_DATA for s in all_statuses) or sample_count == 0:
            return DriftSeverity.INSUFFICIENT_DATA

        # Partial rule: if data & prediction drift are NORMAL and performance is INSUFFICIENT_DATA
        if data_status == DriftSeverity.NORMAL or prediction_status == DriftSeverity.NORMAL:
            return DriftSeverity.NORMAL

        return DriftSeverity.NORMAL

    def extract_derived_alerts(
        self,
        feature_report: FeatureDriftReport,
        prediction_report: PredictionDriftReport,
        performance_report: PerformanceReport,
    ) -> List[MonitoringAlertItem]:
        """
        Extract active WARNING and CRITICAL alerts in-memory across all 3 monitoring components.
        """
        alerts: List[MonitoringAlertItem] = []

        # 1. Feature Drift Alerts
        for f in feature_report.ranked_features:
            if f.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
                obs_val = f.drift_metric_value
                thresh = (
                    self.config.feature_psi_critical
                    if f.severity == DriftSeverity.CRITICAL
                    else self.config.feature_psi_warning
                )
                msg = f"Feature '{f.feature_name}' has {f.severity.value} drift with {f.drift_metric_name}={obs_val:.4f}"
                if f.missing_rate_delta > self.config.missing_rate_delta_critical:
                    msg += f" and missing rate surge (+{f.missing_rate_delta:.1%})"

                alerts.append(
                    MonitoringAlertItem(
                        component="data_drift",
                        feature_name=f.feature_name,
                        severity=f.severity.value,
                        metric_name=f.drift_metric_name.lower(),
                        observed_value=float(obs_val),
                        threshold_value=float(thresh),
                        message=msg,
                    )
                )

        # 2. Prediction Drift Alerts
        # Check model score drift alert
        ms = prediction_report.model_score_drift
        if ms.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
            alerts.append(
                MonitoringAlertItem(
                    component="prediction_drift",
                    feature_name=None,
                    severity=ms.severity.value,
                    metric_name="model_score_psi",
                    observed_value=float(ms.psi_value),
                    threshold_value=float(self.config.prediction_psi_warning),
                    message=ms.alert_message or f"Model score distribution drift {ms.severity.value}",
                )
            )

        # Check risk score drift alert
        rs = prediction_report.risk_score_drift
        if rs.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
            alerts.append(
                MonitoringAlertItem(
                    component="prediction_drift",
                    feature_name=None,
                    severity=rs.severity.value,
                    metric_name="risk_score_psi",
                    observed_value=float(rs.psi_value),
                    threshold_value=float(self.config.prediction_psi_warning),
                    message=rs.alert_message or f"Risk score distribution drift {rs.severity.value}",
                )
            )

        # Check tier drift alert
        td = prediction_report.tier_drift
        if td.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
            alerts.append(
                MonitoringAlertItem(
                    component="prediction_drift",
                    feature_name=None,
                    severity=td.severity.value,
                    metric_name="risk_tier_jsd",
                    observed_value=float(td.jsd_value),
                    threshold_value=float(self.config.tier_jsd_warning),
                    message=td.alert_message or f"Risk tier distribution drift {td.severity.value}",
                )
            )

        # Check action drift alert
        ad = prediction_report.action_drift
        if ad.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
            alerts.append(
                MonitoringAlertItem(
                    component="prediction_drift",
                    feature_name=None,
                    severity=ad.severity.value,
                    metric_name="action_jsd",
                    observed_value=float(ad.jsd_value),
                    threshold_value=float(self.config.action_jsd_warning),
                    message=ad.alert_message or f"Decision action distribution drift {ad.severity.value}",
                )
            )

        # 3. Performance Degradation Alerts
        for da in performance_report.active_degradation_alerts:
            alerts.append(
                MonitoringAlertItem(
                    component="performance",
                    feature_name=None,
                    severity=da.severity.value,
                    metric_name=da.metric_name,
                    observed_value=float(da.observed_value),
                    threshold_value=float(da.baseline_value),
                    message=da.alert_message or f"Performance degradation on {da.metric_name}",
                )
            )

        return alerts

    async def compute_and_persist_snapshot(
        self,
        session: AsyncSession,
        window_type: str,
        window_start: datetime,
        window_end: datetime,
        model_version: str = "1.0.0",
    ) -> ModelMonitoringSnapshot:
        """
        Execute windowed monitoring calculation and idempotently persist a snapshot rollup.
        """
        win_type_upper = window_type.upper().strip()
        if win_type_upper not in ("HOURLY", "DAILY"):
            raise ValueError(f"Persisted snapshots only support 'HOURLY' and 'DAILY', got '{window_type}'.")

        # 1. Extract window data
        df_features, evaluations, cases = await self.extract_monitoring_data(
            session=session,
            start_time=window_start,
            end_time=window_end,
            model_version=model_version,
        )

        sample_count = len(evaluations)
        labeled_count = len([c for c in cases if c.disposition is not None])

        # 2. Run monitoring engines
        feat_rep = self.feature_calculator.compute_feature_drift(
            df=df_features,
            window_type=win_type_upper,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
        )

        pred_rep = self.prediction_calculator.compute_prediction_drift(
            data=evaluations,
            window_type=win_type_upper,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
        )

        perf_rep = self.performance_calculator.compute_performance_report(
            cases=cases,
            window_type=win_type_upper,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
        )

        # 3. Overall health status
        overall_status = self.resolve_overall_health_status(
            data_status=feat_rep.overall_data_drift_status,
            prediction_status=pred_rep.overall_prediction_drift_status,
            perf_status=perf_rep.overall_performance_status,
            sample_count=sample_count,
            labeled_count=labeled_count,
        )

        # 4. Construct snapshot entity
        snapshot = ModelMonitoringSnapshot(
            id=uuid.uuid4(),
            model_version=model_version,
            window_type=win_type_upper,
            window_start=window_start,
            window_end=window_end,
            sample_count=sample_count,
            labeled_count=labeled_count,
            overall_status=overall_status.value,
            data_drift_status=feat_rep.overall_data_drift_status.value,
            prediction_drift_status=pred_rep.overall_prediction_drift_status.value,
            performance_status=perf_rep.overall_performance_status.value,
            feature_drift_summary=feat_rep.to_dict(),
            prediction_drift_summary=pred_rep.to_dict(),
            performance_summary=perf_rep.to_dict(),
            created_at=datetime.now(timezone.utc),
        )

        # 5. Idempotent Upsert & Commit
        repo = MonitoringRepository(session)
        saved = await repo.upsert_snapshot(snapshot)
        await session.commit()

        logger.info(
            f"Persisted monitoring snapshot id={saved.id} [{win_type_upper} {window_start} -> {window_end}] "
            f"status={saved.overall_status} samples={sample_count} labeled={labeled_count}"
        )
        return saved

    # -------------------------------------------------------------------------
    # REST API Service Methods
    # -------------------------------------------------------------------------

    async def get_health_overview(
        self,
        session: AsyncSession,
        window: Optional[str] = "24h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        model_version: str = "1.0.0",
    ) -> MonitoringHealthResponse:
        """
        Evaluate high-level health overview with active alert derivation.
        """
        w_start, w_end, w_type = self.resolve_window_boundaries(window, start_time, end_time)

        repo = MonitoringRepository(session)
        if w_type in ("HOURLY", "DAILY"):
            cached = await repo.get_by_window(model_version, w_type, w_start, w_end)
            if cached is not None:
                feat_rep = FeatureDriftReport.from_dict(cached.feature_drift_summary)
                pred_rep = PredictionDriftReport.from_dict(cached.prediction_drift_summary)
                perf_rep = PerformanceReport.from_dict(cached.performance_summary)
                alerts = self.extract_derived_alerts(feat_rep, pred_rep, perf_rep)

                return MonitoringHealthResponse(
                    model_version=cached.model_version,
                    window_type=cached.window_type,
                    window_start=cached.window_start.isoformat(),
                    window_end=cached.window_end.isoformat(),
                    sample_count=cached.sample_count,
                    labeled_count=cached.labeled_count,
                    overall_status=cached.overall_status,
                    data_drift_status=cached.data_drift_status,
                    prediction_drift_status=cached.prediction_drift_status,
                    performance_status=cached.performance_status,
                    active_alert_count=len(alerts),
                    active_alerts=alerts,
                    created_at=cached.created_at.isoformat(),
                )

        # On-demand calculation
        df_features, evaluations, cases = await self.extract_monitoring_data(
            session=session, start_time=w_start, end_time=w_end, model_version=model_version
        )

        sample_count = len(evaluations)
        labeled_count = len([c for c in cases if c.disposition is not None])

        feat_rep = self.feature_calculator.compute_feature_drift(
            df=df_features, window_type=w_type, window_start=w_start.isoformat(), window_end=w_end.isoformat()
        )
        pred_rep = self.prediction_calculator.compute_prediction_drift(
            data=evaluations, window_type=w_type, window_start=w_start.isoformat(), window_end=w_end.isoformat()
        )
        perf_rep = self.performance_calculator.compute_performance_report(
            cases=cases, window_type=w_type, window_start=w_start.isoformat(), window_end=w_end.isoformat()
        )

        overall_status = self.resolve_overall_health_status(
            feat_rep.overall_data_drift_status,
            pred_rep.overall_prediction_drift_status,
            perf_rep.overall_performance_status,
            sample_count,
            labeled_count,
        )
        alerts = self.extract_derived_alerts(feat_rep, pred_rep, perf_rep)

        return MonitoringHealthResponse(
            model_version=model_version,
            window_type=w_type,
            window_start=w_start.isoformat(),
            window_end=w_end.isoformat(),
            sample_count=sample_count,
            labeled_count=labeled_count,
            overall_status=overall_status.value,
            data_drift_status=feat_rep.overall_data_drift_status.value,
            prediction_drift_status=pred_rep.overall_prediction_drift_status.value,
            performance_status=perf_rep.overall_performance_status.value,
            active_alert_count=len(alerts),
            active_alerts=alerts,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    async def get_feature_drift(
        self,
        session: AsyncSession,
        window: Optional[str] = "24h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        search_term: Optional[str] = None,
        sort_by: str = "psi",
        limit: int = 55,
        offset: int = 0,
        model_version: str = "1.0.0",
    ) -> FeatureDriftListResponse:
        """
        Evaluate 55-feature drift report with filtering, searching, and pagination.
        """
        w_start, w_end, w_type = self.resolve_window_boundaries(window, start_time, end_time)

        df_features, evaluations, _ = await self.extract_monitoring_data(
            session=session, start_time=w_start, end_time=w_end, model_version=model_version
        )

        feat_rep = self.feature_calculator.compute_feature_drift(
            df=df_features, window_type=w_type, window_start=w_start.isoformat(), window_end=w_end.isoformat()
        )

        items: List[FeatureDriftItemResponse] = []
        for f in feat_rep.ranked_features:
            cat = _get_feature_category(f.feature_name)
            if category and cat.lower() != category.lower():
                continue
            if status and f.severity.value.upper() != status.upper():
                continue
            if search_term and search_term.lower() not in f.feature_name.lower():
                continue

            items.append(
                FeatureDriftItemResponse(
                    feature_name=f.feature_name,
                    feature_type=f.feature_type,
                    category=cat,
                    status=f.severity.value,
                    psi=float(f.drift_metric_value if f.drift_metric_name == "PSI" else 0.0),
                    ks_statistic=float(f.ks_statistic) if f.ks_statistic is not None else None,
                    ks_p_value=float(f.ks_pvalue) if f.ks_pvalue is not None else None,
                    js_divergence=float(f.drift_metric_value if f.drift_metric_name == "JSD" else 0.0),
                    missing_rate_current=float(f.observed_missing_rate),
                    missing_rate_delta=float(f.missing_rate_delta),
                    unseen_category_rate=float(f.unseen_category_rate) if f.unseen_category_rate is not None else None,
                )
            )

        # Sort
        sort_key = sort_by.lower().strip()
        if sort_key == "ks_stat":
            items.sort(key=lambda x: x.ks_statistic or 0.0, reverse=True)
        elif sort_key == "missing_delta":
            items.sort(key=lambda x: x.missing_rate_delta, reverse=True)
        else:
            items.sort(key=lambda x: x.psi, reverse=True)

        paginated_items = items[offset : offset + limit]

        return FeatureDriftListResponse(
            model_version=model_version,
            window_type=w_type,
            window_start=w_start.isoformat(),
            window_end=w_end.isoformat(),
            sample_count=len(evaluations),
            overall_status=feat_rep.overall_data_drift_status.value,
            total_features_count=len(feat_rep.ranked_features),
            drifted_features_count=feat_rep.drifted_features_count,
            critical_features_count=feat_rep.critical_features_count,
            warning_features_count=feat_rep.warning_features_count,
            items=paginated_items,
        )

    async def get_feature_drift_detail(
        self,
        feature_name: str,
        session: AsyncSession,
        window: Optional[str] = "24h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        model_version: str = "1.0.0",
    ) -> Optional[FeatureDriftDetailResponse]:
        """
        Evaluate full single-feature drift metrics with baseline vs current bin comparisons.
        """
        w_start, w_end, w_type = self.resolve_window_boundaries(window, start_time, end_time)

        df_features, _, _ = await self.extract_monitoring_data(
            session=session, start_time=w_start, end_time=w_end, model_version=model_version
        )

        feat_rep = self.feature_calculator.compute_feature_drift(
            df=df_features, window_type=w_type, window_start=w_start.isoformat(), window_end=w_end.isoformat()
        )

        target_feat = feat_rep.feature_results.get(feature_name)
        if target_feat is None:
            return None

        cat = _get_feature_category(feature_name)
        base_prof = self.feature_calculator.baseline_profile
        base_dist: Dict[str, float] = {}
        curr_dist: Dict[str, float] = {}
        bin_edges: Optional[List[float]] = None

        if target_feat.feature_type == "numerical" and feature_name in base_prof.features:
            n_prof = base_prof.features[feature_name]
            if isinstance(n_prof, NumericalFeatureProfile):
                bin_edges = n_prof.bin_edges
                if target_feat.expected_proportions:
                    base_dist = {f"bin_{i}": p for i, p in enumerate(target_feat.expected_proportions)}
                if target_feat.observed_proportions:
                    curr_dist = {f"bin_{i}": p for i, p in enumerate(target_feat.observed_proportions)}
        elif target_feat.feature_type == "categorical" and feature_name in base_prof.features:
            c_prof = base_prof.features[feature_name]
            if isinstance(c_prof, CategoricalFeatureProfile):
                base_dist = dict(c_prof.probabilities)
                if target_feat.observed_proportions:
                    curr_dist = {k: p for k, p in zip(c_prof.vocabulary, target_feat.observed_proportions)}

        return FeatureDriftDetailResponse(
            feature_name=target_feat.feature_name,
            feature_type=target_feat.feature_type,
            category=cat,
            status=target_feat.severity.value,
            psi=float(target_feat.drift_metric_value if target_feat.drift_metric_name == "PSI" else 0.0),
            ks_statistic=float(target_feat.ks_statistic) if target_feat.ks_statistic is not None else None,
            ks_p_value=float(target_feat.ks_pvalue) if target_feat.ks_pvalue is not None else None,
            js_divergence=float(target_feat.drift_metric_value if target_feat.drift_metric_name == "JSD" else 0.0),
            missing_rate_baseline=float(target_feat.baseline_missing_rate),
            missing_rate_current=float(target_feat.observed_missing_rate),
            missing_rate_delta=float(target_feat.missing_rate_delta),
            unseen_category_rate=float(target_feat.unseen_category_rate) if target_feat.unseen_category_rate is not None else None,
            unseen_categories=target_feat.unseen_categories or [],
            baseline_distribution=base_dist,
            current_distribution=curr_dist,
            bin_edges=bin_edges,
        )

    async def get_prediction_drift(
        self,
        session: AsyncSession,
        window: Optional[str] = "24h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        model_version: str = "1.0.0",
    ) -> PredictionDriftResponse:
        """
        Evaluate prediction and risk score distribution drift.
        """
        w_start, w_end, w_type = self.resolve_window_boundaries(window, start_time, end_time)

        _, evaluations, _ = await self.extract_monitoring_data(
            session=session, start_time=w_start, end_time=w_end, model_version=model_version
        )

        pred_rep = self.prediction_calculator.compute_prediction_drift(
            data=evaluations, window_type=w_type, window_start=w_start.isoformat(), window_end=w_end.isoformat()
        )

        # Build histogram distributions
        ms_dist: Dict[str, float] = {
            f"bin_{b.bin_index}": b.proportion for b in pred_rep.model_score_drift.observed_bins
        }
        rs_dist: Dict[str, float] = {
            f"bucket_{b.bucket_index}": b.proportion for b in pred_rep.risk_score_drift.observed_buckets
        }
        tier_dist: Dict[str, float] = dict(pred_rep.tier_drift.observed_proportions)
        act_dist: Dict[str, float] = dict(pred_rep.action_drift.observed_proportions)

        active_alerts: List[MonitoringAlertItem] = []
        if pred_rep.model_score_drift.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
            active_alerts.append(
                MonitoringAlertItem(
                    component="prediction_drift",
                    feature_name=None,
                    severity=pred_rep.model_score_drift.severity.value,
                    metric_name="model_score_psi",
                    observed_value=float(pred_rep.model_score_drift.psi_value),
                    threshold_value=float(self.config.prediction_psi_warning),
                    message=pred_rep.model_score_drift.alert_message or "Model score distribution drift",
                )
            )

        return PredictionDriftResponse(
            model_version=model_version,
            window_type=w_type,
            window_start=w_start.isoformat(),
            window_end=w_end.isoformat(),
            sample_count=pred_rep.dataset_row_count,
            overall_status=pred_rep.overall_prediction_drift_status.value,
            model_score_psi=float(pred_rep.model_score_drift.psi_value),
            model_score_status=pred_rep.model_score_drift.severity.value,
            model_score_mean=float(pred_rep.model_score_drift.observed_mean),
            model_score_std=0.0,
            risk_score_psi=float(pred_rep.risk_score_drift.psi_value),
            risk_score_status=pred_rep.risk_score_drift.severity.value,
            risk_tier_jsd=float(pred_rep.tier_drift.jsd_value),
            risk_tier_status=pred_rep.tier_drift.severity.value,
            action_jsd=float(pred_rep.action_drift.jsd_value),
            action_status=pred_rep.action_drift.severity.value,
            override_rate_current=float(pred_rep.override_drift.observed_rate),
            override_rate_baseline=float(pred_rep.override_drift.baseline_rate),
            override_rate_delta=float(pred_rep.override_drift.rate_delta),
            model_score_distribution=ms_dist,
            risk_score_buckets_distribution=rs_dist,
            risk_tier_distribution=tier_dist,
            action_distribution=act_dist,
            active_alerts=active_alerts,
        )

    async def get_performance(
        self,
        session: AsyncSession,
        window: Optional[str] = "24h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operating_threshold: Optional[float] = None,
        model_version: str = "1.0.0",
    ) -> ModelPerformanceResponse:
        """
        Evaluate ground-truth model, operational decision, and review queue performance report.
        """
        w_start, w_end, w_type = self.resolve_window_boundaries(window, start_time, end_time)

        _, _, cases = await self.extract_monitoring_data(
            session=session, start_time=w_start, end_time=w_end, model_version=model_version
        )

        perf_rep = self.performance_calculator.compute_performance_report(
            cases=cases,
            operating_threshold=operating_threshold,
            window_type=w_type,
            window_start=w_start.isoformat(),
            window_end=w_end.isoformat(),
        )

        pm = perf_rep.primary_metrics
        cm = pm.confusion_matrix
        cm_resp = ConfusionMatrixResponse(
            tp=cm.tp, fp=cm.fp, fn=cm.fn, tn=cm.tn, total=cm.total
        )
        prim_resp = ThresholdMetricsResponse(
            threshold=pm.threshold,
            precision=pm.precision,
            recall=pm.recall,
            f1=pm.f1,
            accuracy=pm.accuracy,
            fpr=pm.fpr,
            tpr=pm.tpr,
            confusion_matrix=cm_resp,
        )

        compm = perf_rep.comparison_metrics
        compcm = compm.confusion_matrix
        compcm_resp = ConfusionMatrixResponse(
            tp=compcm.tp, fp=compcm.fp, fn=compcm.fn, tn=compcm.tn, total=compcm.total
        )
        comp_resp = ThresholdMetricsResponse(
            threshold=compm.threshold,
            precision=compm.precision,
            recall=compm.recall,
            f1=compm.f1,
            accuracy=compm.accuracy,
            fpr=compm.fpr,
            tpr=compm.tpr,
            confusion_matrix=compcm_resp,
        )

        ops = perf_rep.operational_metrics
        ops_resp = OperationalMetricsResponse(
            decision_precision_block=ops.decision_precision_block,
            decision_recall_intervention=ops.decision_recall_intervention,
            review_queue_purity=ops.review_queue_purity,
            total_reviews_count=ops.total_reviews_count,
            fraud_in_review_count=ops.fraud_in_review_count,
            total_blocks_count=ops.total_blocks_count,
            fraud_in_block_count=ops.fraud_in_block_count,
        )

        deg_map: Dict[str, MetricDegradationItemResponse] = {}
        for k, v in perf_rep.degradation_results.items():
            deg_map[k] = MetricDegradationItemResponse(
                metric_name=v.metric_name,
                observed_value=float(v.observed_value),
                baseline_value=float(v.baseline_value),
                relative_delta=float(v.relative_delta),
                absolute_delta=float(v.absolute_delta),
                severity=v.severity.value,
                confidence=v.confidence.value,
                alert_message=v.alert_message,
            )

        active_alerts = [
            MonitoringAlertItem(
                component="performance",
                feature_name=None,
                severity=da.severity.value,
                metric_name=da.metric_name,
                observed_value=float(da.observed_value),
                threshold_value=float(da.baseline_value),
                message=da.alert_message or f"Performance degradation on {da.metric_name}",
            )
            for da in perf_rep.active_degradation_alerts
        ]

        return ModelPerformanceResponse(
            model_version=model_version,
            window_type=w_type,
            window_start=w_start.isoformat(),
            window_end=w_end.isoformat(),
            dataset_row_count=perf_rep.dataset_row_count,
            labeled_sample_count=perf_rep.labeled_sample_count,
            fraud_cases_count=perf_rep.fraud_cases_count,
            legitimate_cases_count=perf_rep.legitimate_cases_count,
            suspicious_resolved_count=perf_rep.suspicious_resolved_count,
            overall_performance_status=perf_rep.overall_performance_status.value,
            confidence=perf_rep.confidence.value,
            operating_threshold=perf_rep.operating_threshold,
            primary_metrics=prim_resp,
            comparison_metrics=comp_resp,
            pr_auc=perf_rep.pr_auc,
            roc_auc=perf_rep.roc_auc,
            operational_metrics=ops_resp,
            degradation_results=deg_map,
            active_degradation_alerts=active_alerts,
        )

    async def list_snapshots(
        self,
        session: AsyncSession,
        model_version: Optional[str] = None,
        window_type: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> MonitoringSnapshotListResponse:
        """
        List persisted monitoring snapshots.
        """
        repo = MonitoringRepository(session)
        snapshots, total_count = await repo.list_snapshots(
            model_version=model_version,
            window_type=window_type,
            status=status,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            offset=offset,
        )

        items = [
            MonitoringSnapshotItemResponse(
                id=str(s.id),
                model_version=s.model_version,
                window_type=s.window_type,
                window_start=s.window_start.isoformat(),
                window_end=s.window_end.isoformat(),
                sample_count=s.sample_count,
                labeled_count=s.labeled_count,
                overall_status=s.overall_status,
                data_drift_status=s.data_drift_status,
                prediction_drift_status=s.prediction_drift_status,
                performance_status=s.performance_status,
                created_at=s.created_at.isoformat(),
            )
            for s in snapshots
        ]

        return MonitoringSnapshotListResponse(
            total_count=total_count,
            limit=limit,
            offset=offset,
            items=items,
        )


_monitoring_service_instance: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """Dependency provider returning singleton MonitoringService instance."""
    global _monitoring_service_instance
    if _monitoring_service_instance is None:
        _monitoring_service_instance = MonitoringService()
    return _monitoring_service_instance
