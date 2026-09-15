"""
Audit Log Repository for Persisting System Events & Telemetry.

Provides asynchronous access for `AuditLog` ORM entities with strict
transaction-boundary neutrality and append-oriented semantics.
"""

from typing import List, Optional, Sequence, Union
import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditEntityType
from backend.app.repositories.exceptions import PersistenceError


class AuditLogRepository:
    """
    Asynchronous repository for appending and querying `AuditLog` entities.

    Design Principles:
    - Session Injection: Operates strictly on an injected AsyncSession.
    - Transaction Neutrality: Does not commit, rollback, or flush implicitly.
    - Append-Oriented: Only exposes query and append operations; no update or delete
      methods are provided at the repository layer.
      (Note: Full immutability guarantees require database-level privilege controls).
    - Deterministic Ordering: Queries return audit events newest-first by event_timestamp
      descending, then id descending.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the repository with an active AsyncSession.

        Args:
            session: Active SQLAlchemy asynchronous database session.
        """
        self._session = session

    async def get_by_id(
        self,
        audit_log_id: uuid.UUID,
    ) -> Optional[AuditLog]:
        """
        Query an AuditLog record by its primary key UUID.

        Args:
            audit_log_id: Primary key UUID of the audit log record.

        Returns:
            The AuditLog entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(AuditLog).where(AuditLog.id == audit_log_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve audit log by ID '{audit_log_id}': {exc}"
            ) from exc

    async def list_by_entity(
        self,
        entity_type: Union[AuditEntityType, str],
        entity_id: Optional[uuid.UUID] = None,
        limit: Optional[int] = None,
    ) -> List[AuditLog]:
        """
        Retrieve audit log records for a given entity type and optional entity ID.

        Results are ordered deterministically by event_timestamp descending, then id descending.

        Args:
            entity_type: AuditEntityType enum or string representation.
            entity_id: Optional UUID identifier of the target entity.
            limit: Optional maximum number of log records to return (must be > 0).

        Returns:
            List of AuditLog entities matching the criteria.

        Raises:
            ValueError: If limit is not None and <= 0.
            PersistenceError: If an unexpected database query failure occurs.
        """
        if limit is not None and limit <= 0:
            raise ValueError(f"limit must be a positive integer, got {limit}")

        try:
            # Handle enum or string input
            etype = (
                AuditEntityType(entity_type)
                if isinstance(entity_type, str) and not isinstance(entity_type, AuditEntityType)
                else entity_type
            )
            stmt = select(AuditLog).where(AuditLog.entity_type == etype)
            if entity_id is not None:
                stmt = stmt.where(AuditLog.entity_id == entity_id)

            stmt = stmt.order_by(AuditLog.event_timestamp.desc(), AuditLog.id.desc())

            if limit is not None:
                stmt = stmt.limit(limit)

            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list audit logs for entity '{entity_type}:{entity_id}': {exc}"
            ) from exc

    async def add(
        self,
        audit_log: AuditLog,
    ) -> AuditLog:
        """
        Attach a single AuditLog record to the session.

        Does not commit or flush; transaction lifecycle is owned by the caller.

        Args:
            audit_log: AuditLog instance to stage.

        Returns:
            The same AuditLog instance attached to the session.
        """
        self._session.add(audit_log)
        return audit_log

    async def add_many(
        self,
        audit_logs: Sequence[AuditLog],
    ) -> List[AuditLog]:
        """
        Attach multiple AuditLog instances to the session in batch.

        If the input sequence is empty, immediately returns an empty list without
        issuing any database operations.

        Args:
            audit_logs: Sequence of AuditLog instances.

        Returns:
            List of the attached AuditLog instances preserving input order.
        """
        if not audit_logs:
            return []

        log_list = list(audit_logs)
        self._session.add_all(log_list)
        return log_list


__all__ = ["AuditLogRepository"]
