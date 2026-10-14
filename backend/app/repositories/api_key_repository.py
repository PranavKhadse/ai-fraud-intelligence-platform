"""
APIKey Repository for Machine-to-Machine Credentials Persistence.

Provides asynchronous access for querying, validating, and staging `APIKey`
ORM entities with strict transaction-boundary neutrality.
"""

from datetime import datetime, timezone
from typing import List, Optional, Sequence
import uuid

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.api_key import APIKey
from backend.app.repositories.exceptions import PersistenceError


class APIKeyRepository:
    """
    Asynchronous repository for managing APIKey ORM entities.

    Design Principles:
    - Session Injection: Operates strictly on an injected AsyncSession.
    - Transaction Boundary Neutrality: Never commits, flushes, or rolls back implicitly.
    - Exception Wrapping: Translates database driver exceptions into domain PersistenceError.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the APIKey repository.

        Args:
            session: Active SQLAlchemy asynchronous database session.
        """
        self._session = session

    async def get_by_id(
        self,
        key_id: uuid.UUID,
    ) -> Optional[APIKey]:
        """
        Retrieve an API key record by primary key UUID.

        Args:
            key_id: UUID primary key.

        Returns:
            The APIKey entity if found, otherwise None.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = select(APIKey).where(APIKey.id == key_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve API key by ID '{key_id}': {exc}"
            ) from exc

    async def get_by_hashed_key(
        self,
        hashed_key: str,
    ) -> Optional[APIKey]:
        """
        Retrieve an API key record by its exact cryptographic hash digest.

        Args:
            hashed_key: Cryptographic hash of the secret key token.

        Returns:
            The APIKey entity if found, otherwise None.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = select(APIKey).where(APIKey.hashed_key == hashed_key)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve API key by hashed key: {exc}"
            ) from exc

    async def list_by_client_id(
        self,
        client_id: str,
        active_only: bool = False,
    ) -> List[APIKey]:
        """
        Retrieve all API keys associated with a specific client identifier.

        Args:
            client_id: Client identifier string.
            active_only: If True, only returns active non-revoked keys.

        Returns:
            List of matching APIKey entities.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = select(APIKey).where(APIKey.client_id == client_id)
            if active_only:
                stmt = stmt.where(APIKey.is_active == True)  # noqa: E712
            stmt = stmt.order_by(APIKey.created_at.desc(), APIKey.id.desc())

            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list API keys for client '{client_id}': {exc}"
            ) from exc

    async def list_by_prefix(
        self,
        key_prefix: str,
        active_only: bool = True,
    ) -> List[APIKey]:
        """
        Retrieve API keys matching a public key prefix (e.g. 'pk_live_').

        Args:
            key_prefix: Public prefix string.
            active_only: If True, only returns active non-revoked keys.

        Returns:
            List of matching APIKey entities.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = select(APIKey).where(APIKey.key_prefix == key_prefix)
            if active_only:
                stmt = stmt.where(APIKey.is_active == True)  # noqa: E712
            stmt = stmt.order_by(APIKey.created_at.desc(), APIKey.id.desc())

            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list API keys by prefix '{key_prefix}': {exc}"
            ) from exc

    async def list_by_user_id(
        self,
        user_id: uuid.UUID,
    ) -> List[APIKey]:
        """
        Retrieve all API keys owned by a specific User account.

        Args:
            user_id: Owning User UUID.

        Returns:
            List of APIKey entities ordered newest-first.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = (
                select(APIKey)
                .where(APIKey.user_id == user_id)
                .order_by(APIKey.created_at.desc(), APIKey.id.desc())
            )
            result = await self._session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list API keys for user '{user_id}': {exc}"
            ) from exc

    async def add(
        self,
        api_key: APIKey,
    ) -> APIKey:
        """
        Attach a new APIKey instance to the session.

        Does not commit or flush; transaction boundary is caller-owned.

        Args:
            api_key: APIKey entity to attach.

        Returns:
            The attached APIKey entity.
        """
        self._session.add(api_key)
        return api_key

    async def add_many(
        self,
        api_keys: Sequence[APIKey],
    ) -> List[APIKey]:
        """
        Attach multiple APIKey instances to the session in batch.

        Args:
            api_keys: Sequence of APIKey entities.

        Returns:
            List of attached APIKey entities preserving order.
        """
        if not api_keys:
            return []
        key_list = list(api_keys)
        self._session.add_all(key_list)
        return key_list

    async def update_last_used(
        self,
        key_id: uuid.UUID,
        used_time: Optional[datetime] = None,
    ) -> Optional[APIKey]:
        """
        Update the last_used_at timestamp for a specified API key.

        Args:
            key_id: Target APIKey UUID.
            used_time: Optional timestamp (defaults to current UTC time).

        Returns:
            The updated APIKey entity if found, otherwise None.

        Raises:
            PersistenceError: On database query failure.
        """
        api_key = await self.get_by_id(key_id)
        if not api_key:
            return None
        api_key.last_used_at = used_time or datetime.now(timezone.utc)
        return api_key


__all__ = ["APIKeyRepository"]
