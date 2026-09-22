"""
Monitoring Snapshot Repository for Phase 13 ML & Model Monitoring.

Provides data access and persistence operations for `model_monitoring_snapshots`:
- Idempotent upsert of monitoring snapshots.
- Point-in-time and window-range snapshot retrieval.
- Paginated listing with filtering by model version, window type, status, and dates.
- Bounded retention cleanup of expired hourly and daily snapshots without affecting business tables.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence, Tuple
import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.monitoring_snapshot import ModelMonitoringSnapshot


class MonitoringRepository:
    """
    Asynchronous repository managing ModelMonitoringSnapshot database persistence.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the repository with an active AsyncSession.

        Args:
            session: Active SQLAlchemy asynchronous database session.
        """
        self._session = session

    async def get_by_id(self, snapshot_id: uuid.UUID) -> Optional[ModelMonitoringSnapshot]:
        """Retrieve a monitoring snapshot by its primary key UUID."""
        stmt = select(ModelMonitoringSnapshot).where(ModelMonitoringSnapshot.id == snapshot_id)
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def get_by_window(
        self,
        model_version: str,
        window_type: str,
        window_start: datetime,
        window_end: datetime,
    ) -> Optional[ModelMonitoringSnapshot]:
        """
        Retrieve a snapshot for an exact model version, window type, and time interval.
        """
        stmt = (
            select(ModelMonitoringSnapshot)
            .where(
                ModelMonitoringSnapshot.model_version == model_version,
                ModelMonitoringSnapshot.window_type == window_type,
                ModelMonitoringSnapshot.window_start == window_start,
                ModelMonitoringSnapshot.window_end == window_end,
            )
        )
        result = await self._session.execute(stmt)
        return result.scalars().first()

    async def upsert_snapshot(
        self,
        snapshot: ModelMonitoringSnapshot,
    ) -> ModelMonitoringSnapshot:
        """
        Idempotently insert or update a model monitoring snapshot.

        Enforces the unique constraint on (model_version, window_type, window_start, window_end).
        If a matching snapshot exists, updates all diagnostic summary fields, sample counts,
        and statuses in-place.
        """
        existing = await self.get_by_window(
            model_version=snapshot.model_version,
            window_type=snapshot.window_type,
            window_start=snapshot.window_start,
            window_end=snapshot.window_end,
        )

        if existing is not None:
            existing.sample_count = snapshot.sample_count
            existing.labeled_count = snapshot.labeled_count
            existing.overall_status = snapshot.overall_status
            existing.data_drift_status = snapshot.data_drift_status
            existing.prediction_drift_status = snapshot.prediction_drift_status
            existing.performance_status = snapshot.performance_status
            existing.feature_drift_summary = snapshot.feature_drift_summary
            existing.prediction_drift_summary = snapshot.prediction_drift_summary
            existing.performance_summary = snapshot.performance_summary
            existing.created_at = snapshot.created_at or datetime.now(timezone.utc)
            await self._session.flush()
            return existing

        self._session.add(snapshot)
        await self._session.flush()
        return snapshot

    async def list_snapshots(
        self,
        model_version: Optional[str] = None,
        window_type: Optional[str] = None,
        status: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[ModelMonitoringSnapshot], int]:
        """
        Query paginated historical monitoring snapshots with flexible filtering.

        Returns:
            Tuple of (list_of_snapshots, total_matching_count).
        """
        stmt = select(ModelMonitoringSnapshot)
        count_stmt = select(func.count(ModelMonitoringSnapshot.id))

        if model_version:
            stmt = stmt.where(ModelMonitoringSnapshot.model_version == model_version)
            count_stmt = count_stmt.where(ModelMonitoringSnapshot.model_version == model_version)

        if window_type:
            stmt = stmt.where(ModelMonitoringSnapshot.window_type == window_type)
            count_stmt = count_stmt.where(ModelMonitoringSnapshot.window_type == window_type)

        if status:
            stmt = stmt.where(ModelMonitoringSnapshot.overall_status == status)
            count_stmt = count_stmt.where(ModelMonitoringSnapshot.overall_status == status)

        if start_time:
            stmt = stmt.where(ModelMonitoringSnapshot.window_start >= start_time)
            count_stmt = count_stmt.where(ModelMonitoringSnapshot.window_start >= start_time)

        if end_time:
            stmt = stmt.where(ModelMonitoringSnapshot.window_end <= end_time)
            count_stmt = count_stmt.where(ModelMonitoringSnapshot.window_end <= end_time)

        # Order by window_end descending (most recent first)
        stmt = stmt.order_by(ModelMonitoringSnapshot.window_end.desc()).limit(limit).offset(offset)

        total_res = await self._session.execute(count_stmt)
        total_count = total_res.scalar() or 0

        rows_res = await self._session.execute(stmt)
        snapshots = list(rows_res.scalars().all())

        return snapshots, total_count

    async def cleanup_expired_snapshots(
        self,
        hourly_retention_days: int = 30,
        daily_retention_days: int = 365,
        now: Optional[datetime] = None,
    ) -> int:
        """
        Enforce bounded retention policy on monitoring snapshots:
        - Deletes HOURLY snapshots older than `hourly_retention_days`.
        - Deletes DAILY snapshots older than `daily_retention_days`.

        Guarantees:
        - Deletes strictly from `model_monitoring_snapshots`.
        - Zero mutations to `transactions`, `risk_evaluations`, `cases`, or `audit_logs`.

        Returns:
            Total count of deleted snapshot records.
        """
        current_time = now or datetime.now(timezone.utc)
        hourly_cutoff = current_time - timedelta(days=hourly_retention_days)
        daily_cutoff = current_time - timedelta(days=daily_retention_days)

        stmt = delete(ModelMonitoringSnapshot).where(
            (
                (ModelMonitoringSnapshot.window_type == "HOURLY")
                & (ModelMonitoringSnapshot.window_end < hourly_cutoff)
            )
            | (
                (ModelMonitoringSnapshot.window_type == "DAILY")
                & (ModelMonitoringSnapshot.window_end < daily_cutoff)
            )
        )

        result = await self._session.execute(stmt)
        await self._session.flush()
        return result.rowcount or 0
