"""
Reason Code Repository for Persisting Human-Readable Explanation Drivers.

Provides asynchronous access for `EvaluationReasonCode` ORM entities with strict
transaction-boundary neutrality.
"""

from typing import List, Optional, Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.reason_code import EvaluationReasonCode
from backend.app.repositories.exceptions import PersistenceError


class ReasonCodeRepository:
    """
    Asynchronous repository for querying and staging `EvaluationReasonCode` entities.

    Design Principles:
    - Session Injection: Operates on an injected AsyncSession.
    - Deterministic Ordering: Queries return reason codes ordered by rank ascending.
    - Batch Operations: `add_many` stages multiple records safely without committing.
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
        reason_id: uuid.UUID,
    ) -> Optional[EvaluationReasonCode]:
        """
        Query an EvaluationReasonCode by its primary key UUID.

        Args:
            reason_id: Primary key UUID of the reason code record.

        Returns:
            The EvaluationReasonCode entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(EvaluationReasonCode).where(EvaluationReasonCode.id == reason_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve reason code by ID '{reason_id}': {exc}"
            ) from exc

    async def list_by_evaluation_id(
        self,
        evaluation_id: uuid.UUID,
    ) -> List[EvaluationReasonCode]:
        """
        Retrieve all reason codes for an evaluation ordered by presentation rank.

        Results are ordered deterministically by rank ascending, then id ascending.

        Args:
            evaluation_id: Foreign key UUID referencing the parent RiskEvaluation.

        Returns:
            List of EvaluationReasonCode entities.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(EvaluationReasonCode)
                .where(EvaluationReasonCode.evaluation_id == evaluation_id)
                .order_by(EvaluationReasonCode.rank.asc(), EvaluationReasonCode.id.asc())
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list reason codes for evaluation '{evaluation_id}': {exc}"
            ) from exc

    async def add(
        self,
        reason_code: EvaluationReasonCode,
    ) -> EvaluationReasonCode:
        """
        Attach a single EvaluationReasonCode to the session.

        Args:
            reason_code: EvaluationReasonCode instance to stage.

        Returns:
            The same EvaluationReasonCode instance attached to the session.
        """
        self._session.add(reason_code)
        return reason_code

    async def add_many(
        self,
        reason_codes: Sequence[EvaluationReasonCode],
    ) -> List[EvaluationReasonCode]:
        """
        Attach multiple EvaluationReasonCode instances to the session in batch.

        If the input sequence is empty, immediately returns an empty list without
        issuing any database operations.

        Args:
            reason_codes: Sequence of EvaluationReasonCode instances.

        Returns:
            List of the attached EvaluationReasonCode instances preserving input order.
        """
        if not reason_codes:
            return []

        code_list = list(reason_codes)
        self._session.add_all(code_list)
        return code_list


# Re-export with explicit alias
EvaluationReasonCodeRepository = ReasonCodeRepository

__all__ = ["ReasonCodeRepository", "EvaluationReasonCodeRepository"]
