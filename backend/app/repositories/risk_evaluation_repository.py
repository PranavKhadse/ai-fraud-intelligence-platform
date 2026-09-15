"""
Risk Evaluation Repository for Persisting Evaluator Decisions & Provenance.

Provides asynchronous access for `RiskEvaluation` ORM entities with strict
transaction-boundary neutrality.
"""

from typing import List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.risk_evaluation import RiskEvaluation
from backend.app.repositories.exceptions import PersistenceError


class RiskEvaluationRepository:
    """
    Asynchronous repository for querying and staging `RiskEvaluation` ORM entities.

    Design Principles:
    - Session Injection: Receives an active AsyncSession; never manages session lifecycles.
    - Transaction Neutrality: Does not commit, rollback, or flush implicitly.
    - Provenance Retrieval: Supports deterministic list and latest-evaluation queries per transaction.
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
        evaluation_id: uuid.UUID,
    ) -> Optional[RiskEvaluation]:
        """
        Query a RiskEvaluation by its primary key UUID.

        Args:
            evaluation_id: Internal UUID primary key of the evaluation.

        Returns:
            The RiskEvaluation entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(RiskEvaluation).where(RiskEvaluation.id == evaluation_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve risk evaluation by ID '{evaluation_id}': {exc}"
            ) from exc

    async def get_by_transaction_id(
        self,
        transaction_id: uuid.UUID,
    ) -> Optional[RiskEvaluation]:
        """
        Retrieve the latest RiskEvaluation associated with a Transaction.

        Because a transaction may undergo re-evaluations or policy simulations,
        this method returns the most recent evaluation by evaluated_at descending.

        Args:
            transaction_id: Foreign key UUID referencing the parent Transaction.

        Returns:
            The most recent RiskEvaluation entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(RiskEvaluation)
                .where(RiskEvaluation.transaction_id == transaction_id)
                .order_by(RiskEvaluation.evaluated_at.desc(), RiskEvaluation.id.desc())
                .limit(1)
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve latest risk evaluation for transaction '{transaction_id}': {exc}"
            ) from exc

    async def list_by_transaction_id(
        self,
        transaction_id: uuid.UUID,
    ) -> List[RiskEvaluation]:
        """
        Retrieve all RiskEvaluations for a Transaction ordered deterministically.

        Results are ordered newest-first by evaluated_at descending, then id descending.

        Args:
            transaction_id: Foreign key UUID referencing the parent Transaction.

        Returns:
            List of RiskEvaluation entities associated with the transaction.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(RiskEvaluation)
                .where(RiskEvaluation.transaction_id == transaction_id)
                .order_by(RiskEvaluation.evaluated_at.desc(), RiskEvaluation.id.desc())
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list risk evaluations for transaction '{transaction_id}': {exc}"
            ) from exc

    async def add(
        self,
        evaluation: RiskEvaluation,
    ) -> RiskEvaluation:
        """
        Attach a new RiskEvaluation entity to the session for persistence.

        Does not commit or flush; transaction lifecycle is owned by the caller.

        Args:
            evaluation: RiskEvaluation instance to stage.

        Returns:
            The same RiskEvaluation instance attached to the session.
        """
        self._session.add(evaluation)
        return evaluation


__all__ = ["RiskEvaluationRepository"]
