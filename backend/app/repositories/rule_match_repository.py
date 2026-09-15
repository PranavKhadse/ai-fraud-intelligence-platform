"""
Rule Match Repository for Persisting Triggered Business Rule Outcomes.

Provides asynchronous access for `EvaluationRuleMatch` ORM entities with strict
transaction-boundary neutrality.
"""

from typing import List, Optional, Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.rule_match import EvaluationRuleMatch
from backend.app.repositories.exceptions import PersistenceError


class RuleMatchRepository:
    """
    Asynchronous repository for querying and staging `EvaluationRuleMatch` entities.

    Design Principles:
    - Session Injection: Operates on an injected AsyncSession.
    - Deterministic Ordering: Queries return rule matches ordered by evaluation priority ascending.
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
        match_id: uuid.UUID,
    ) -> Optional[EvaluationRuleMatch]:
        """
        Query an EvaluationRuleMatch by its primary key UUID.

        Args:
            match_id: Primary key UUID of the rule match.

        Returns:
            The EvaluationRuleMatch entity if found, otherwise None.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = select(EvaluationRuleMatch).where(EvaluationRuleMatch.id == match_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve rule match by ID '{match_id}': {exc}"
            ) from exc

    async def list_by_evaluation_id(
        self,
        evaluation_id: uuid.UUID,
    ) -> List[EvaluationRuleMatch]:
        """
        Retrieve all triggered business rules for an evaluation ordered by priority.

        Results are ordered deterministically by priority ascending, then id ascending.

        Args:
            evaluation_id: Foreign key UUID referencing the parent RiskEvaluation.

        Returns:
            List of EvaluationRuleMatch entities.

        Raises:
            PersistenceError: If an unexpected database query failure occurs.
        """
        try:
            stmt = (
                select(EvaluationRuleMatch)
                .where(EvaluationRuleMatch.evaluation_id == evaluation_id)
                .order_by(EvaluationRuleMatch.priority.asc(), EvaluationRuleMatch.id.asc())
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list rule matches for evaluation '{evaluation_id}': {exc}"
            ) from exc

    async def add(
        self,
        rule_match: EvaluationRuleMatch,
    ) -> EvaluationRuleMatch:
        """
        Attach a single EvaluationRuleMatch to the session.

        Args:
            rule_match: EvaluationRuleMatch instance to stage.

        Returns:
            The same EvaluationRuleMatch instance attached to the session.
        """
        self._session.add(rule_match)
        return rule_match

    async def add_many(
        self,
        rule_matches: Sequence[EvaluationRuleMatch],
    ) -> List[EvaluationRuleMatch]:
        """
        Attach multiple EvaluationRuleMatch instances to the session in batch.

        If the input sequence is empty, immediately returns an empty list without
        issuing any database operations.

        Args:
            rule_matches: Sequence of EvaluationRuleMatch instances.

        Returns:
            List of the attached EvaluationRuleMatch instances preserving input order.
        """
        if not rule_matches:
            return []

        match_list = list(rule_matches)
        self._session.add_all(match_list)
        return match_list


# Re-export with explicit alias
EvaluationRuleMatchRepository = RuleMatchRepository

__all__ = ["RuleMatchRepository", "EvaluationRuleMatchRepository"]
