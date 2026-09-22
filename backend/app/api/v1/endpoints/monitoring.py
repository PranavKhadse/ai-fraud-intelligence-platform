"""
Monitoring REST API Endpoints for Phase 13 ML & Model Monitoring.

Provides public diagnostic endpoints for:
- Monitoring health summary & derived active alerts
- 55-feature data drift report and single-feature diagnostic inspections
- Prediction, risk score bucket, risk tier, and decision action drift
- Ground-truth model classification, operational decision performance, and review queue purity
- Persisted historical monitoring snapshot rollup listings
"""

from datetime import datetime
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_db_session
from backend.app.schemas.monitoring import (
    FeatureDriftDetailResponse,
    FeatureDriftListResponse,
    ModelPerformanceResponse,
    MonitoringHealthResponse,
    MonitoringSnapshotListResponse,
    PredictionDriftResponse,
)
from backend.app.services.monitoring_service import (
    MonitoringService,
    get_monitoring_service,
)

logger = logging.getLogger("fraud_api.monitoring")

router = APIRouter(prefix="/monitoring", tags=["Model Monitoring"])


@router.get(
    "/health",
    response_model=MonitoringHealthResponse,
    summary="Get Model Monitoring Health & Active Alerts",
    description=(
        "Retrieve overall monitoring health status, individual component statuses "
        "(feature drift, prediction drift, performance), sample counts, and active alerts."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_monitoring_health(
    window: Optional[str] = Query("24h", description="Time window ('1h', '24h', '7d', '30d', 'custom')"),
    start_time: Optional[datetime] = Query(None, description="ISO UTC start timestamp for custom window"),
    end_time: Optional[datetime] = Query(None, description="ISO UTC end timestamp for custom window"),
    model_version: str = Query("1.0.0", description="Target model version"),
    session: AsyncSession = Depends(get_db_session),
    service: MonitoringService = Depends(get_monitoring_service),
) -> MonitoringHealthResponse:
    """Retrieve overall system monitoring health."""
    return await service.get_health_overview(
        session=session,
        window=window,
        start_time=start_time,
        end_time=end_time,
        model_version=model_version,
    )


@router.get(
    "/drift/features",
    response_model=FeatureDriftListResponse,
    summary="Get Ranked Feature Drift Report",
    description=(
        "Retrieve ranked drift metrics across all 55 canonical features including "
        "PSI, two-sample KS statistics, categorical JSD, and missing-rate deltas."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_feature_drift_report(
    window: Optional[str] = Query("24h", description="Time window ('1h', '24h', '7d', '30d', 'custom')"),
    start_time: Optional[datetime] = Query(None, description="ISO UTC start timestamp for custom window"),
    end_time: Optional[datetime] = Query(None, description="ISO UTC end timestamp for custom window"),
    category: Optional[str] = Query(None, description="Filter by feature category group"),
    status: Optional[str] = Query(None, description="Filter by drift status ('NORMAL', 'WARNING', 'CRITICAL')"),
    search_term: Optional[str] = Query(None, description="Search feature names by substring"),
    sort_by: str = Query("psi", description="Sort by 'psi', 'ks_stat', or 'missing_delta'"),
    limit: int = Query(55, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    model_version: str = Query("1.0.0", description="Target model version"),
    session: AsyncSession = Depends(get_db_session),
    service: MonitoringService = Depends(get_monitoring_service),
) -> FeatureDriftListResponse:
    """Retrieve 55-feature drift report."""
    return await service.get_feature_drift(
        session=session,
        window=window,
        start_time=start_time,
        end_time=end_time,
        category=category,
        status=status,
        search_term=search_term,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
        model_version=model_version,
    )


@router.get(
    "/drift/features/{feature_name}",
    response_model=FeatureDriftDetailResponse,
    summary="Get Detailed Single Feature Drift Inspection",
    description="Retrieve comprehensive drift diagnostics and distribution histogram comparisons for a single feature.",
    status_code=status.HTTP_200_OK,
)
async def get_single_feature_drift(
    feature_name: str = Path(..., description="Canonical feature name"),
    window: Optional[str] = Query("24h", description="Time window"),
    start_time: Optional[datetime] = Query(None, description="ISO UTC start timestamp"),
    end_time: Optional[datetime] = Query(None, description="ISO UTC end timestamp"),
    model_version: str = Query("1.0.0", description="Target model version"),
    session: AsyncSession = Depends(get_db_session),
    service: MonitoringService = Depends(get_monitoring_service),
) -> FeatureDriftDetailResponse:
    """Retrieve single-feature drift breakdown."""
    detail = await service.get_feature_drift_detail(
        feature_name=feature_name,
        session=session,
        window=window,
        start_time=start_time,
        end_time=end_time,
        model_version=model_version,
    )
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature '{feature_name}' not found in canonical 55-feature baseline profile.",
        )
    return detail


@router.get(
    "/drift/predictions",
    response_model=PredictionDriftResponse,
    summary="Get Prediction & Risk Score Drift",
    description=(
        "Retrieve continuous model score PSI, 10-bucket risk score distribution drift, "
        "risk tier shifts, and operational decision action volume changes."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_prediction_drift(
    window: Optional[str] = Query("24h", description="Time window"),
    start_time: Optional[datetime] = Query(None, description="ISO UTC start timestamp"),
    end_time: Optional[datetime] = Query(None, description="ISO UTC end timestamp"),
    model_version: str = Query("1.0.0", description="Target model version"),
    session: AsyncSession = Depends(get_db_session),
    service: MonitoringService = Depends(get_monitoring_service),
) -> PredictionDriftResponse:
    """Retrieve prediction and risk score distribution drift."""
    return await service.get_prediction_drift(
        session=session,
        window=window,
        start_time=start_time,
        end_time=end_time,
        model_version=model_version,
    )


@router.get(
    "/performance",
    response_model=ModelPerformanceResponse,
    summary="Get Ground-Truth Performance & Queue Purity",
    description=(
        "Retrieve pure model classification performance at threshold 0.78, hybrid operational "
        "decision metrics, human review queue purity, and mathematical degradation alerts."
    ),
    status_code=status.HTTP_200_OK,
)
async def get_model_performance(
    window: Optional[str] = Query("24h", description="Time window"),
    start_time: Optional[datetime] = Query(None, description="ISO UTC start timestamp"),
    end_time: Optional[datetime] = Query(None, description="ISO UTC end timestamp"),
    operating_threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Custom operating threshold (default: 0.78)"),
    model_version: str = Query("1.0.0", description="Target model version"),
    session: AsyncSession = Depends(get_db_session),
    service: MonitoringService = Depends(get_monitoring_service),
) -> ModelPerformanceResponse:
    """Retrieve ground-truth performance and degradation report."""
    return await service.get_performance(
        session=session,
        window=window,
        start_time=start_time,
        end_time=end_time,
        operating_threshold=operating_threshold,
        model_version=model_version,
    )


@router.get(
    "/snapshots",
    response_model=MonitoringSnapshotListResponse,
    summary="List Persisted Historical Monitoring Snapshots",
    description="Retrieve paginated historical hourly and daily monitoring rollups for trend visualization.",
    status_code=status.HTTP_200_OK,
)
async def list_persisted_snapshots(
    model_version: Optional[str] = Query(None, description="Filter by model version"),
    window_type: Optional[str] = Query(None, description="Filter by window type ('HOURLY', 'DAILY')"),
    status: Optional[str] = Query(None, description="Filter by overall status"),
    start_time: Optional[datetime] = Query(None, description="Filter by window start timestamp"),
    end_time: Optional[datetime] = Query(None, description="Filter by window end timestamp"),
    limit: int = Query(50, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    session: AsyncSession = Depends(get_db_session),
    service: MonitoringService = Depends(get_monitoring_service),
) -> MonitoringSnapshotListResponse:
    """List historical persisted snapshots."""
    return await service.list_snapshots(
        session=session,
        model_version=model_version,
        window_type=window_type,
        status=status,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset,
    )
