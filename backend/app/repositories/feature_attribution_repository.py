"""
Feature Attribution Repository for Persisting Local TreeSHAP Explanations.

Provides asynchronous access for `EvaluationFeatureAttribution` ORM entities with strict
transaction-boundary neutrality.
"""

from typing import List, Optional, Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.feature_attribution import EvaluationFeatureAttribution
from backend.app.repositories.exceptions import PersistenceError


class FeatureAttributionRepository:
    """
    Asynchronous repository for querying and staging `EvaluationFeatureAttribution` entities.

    Design Principles:
    - Session Injection: Operates on an injected AsyncSession.
    - Deterministic Ordering: Queries return feature attributions ordered by importance rank ascending.
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
        attribution_id: uuid.UUID,
    ) -> Optional[EvaluationFeatureAttribution]:
        """
        Query an EvaluationFeatureAttribution by its primary key UUID.

        Args:
            attribution_id: Primary key UUID of the attribution record.

        Returns:
            The EvaluationFeatureAttribution entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(EvaluationFeatureAttribution).where(
                EvaluationFeatureAttribution.id == attribution_id
            )
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve feature attribution by ID '{attribution_id}': {exc}"
            ) from exc

    async def list_by_evaluation_id(
        self,
        evaluation_id: uuid.UUID,
    ) -> List[EvaluationFeatureAttribution]:
        """
        Retrieve all feature attributions for an evaluation ordered by importance rank.

        Results are ordered deterministically by rank ascending, then id ascending.

        Args:
            evaluation_id: Foreign key UUID referencing the parent RiskEvaluation.

        Returns:
            List of EvaluationFeatureAttribution entities.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(EvaluationFeatureAttribution)
                .where(EvaluationFeatureAttribution.evaluation_id == evaluation_id)
                .order_by(
                    EvaluationFeatureAttribution.rank.asc(),
                    EvaluationFeatureAttribution.id.asc(),
                )
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list feature attributions for evaluation '{evaluation_id}': {exc}"
            ) from exc

    async def add(
        self,
        feature_attribution: EvaluationFeatureAttribution,
    ) -> EvaluationFeatureAttribution:
        """
        Attach a single EvaluationFeatureAttribution to the session.

        Args:
            feature_attribution: EvaluationFeatureAttribution instance to stage.

        Returns:
            The same EvaluationFeatureAttribution instance attached to the session.
        """
        self._session.add(feature_attribution)
        return feature_attribution

    async def add_many(
        self,
        feature_attributions: Sequence[EvaluationFeatureAttribution],
    ) -> List[EvaluationFeatureAttribution]:
        """
        Attach multiple EvaluationFeatureAttribution instances to the session in batch.

        If the input sequence is empty, immediately returns an empty list without
        issuing any database operations.

        Args:
            feature_attributions: Sequence of EvaluationFeatureAttribution instances.

        Returns:
            List of the attached EvaluationFeatureAttribution instances preserving input order.
        """
        if not feature_attributions:
            return []

        attr_list = list(feature_attributions)
        self._session.add_all(attr_list)
        return attr_list


# Re-export with explicit alias
EvaluationFeatureAttributionRepository = FeatureAttributionRepository

__all__ = ["FeatureAttributionRepository", "EvaluationFeatureAttributionRepository"]
