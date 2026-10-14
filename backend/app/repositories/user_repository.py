"""
User Repository for User Identity, Authentication & RBAC Persistence.

Provides asynchronous CRUD access, username/email uniqueness lookups,
and user listing with role/status filtering under external transaction boundaries.
"""

from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple, Union
import uuid

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.enums import AuditActorType, UserRole
from backend.app.db.models.user import User
from backend.app.repositories.exceptions import (
    PersistenceConflictError,
    PersistenceError,
)


class UserRepository:
    """
    Asynchronous repository for managing User ORM entities.

    Design Principles:
    - Session Injection: Operates strictly on an injected AsyncSession.
    - Transaction Boundary Neutrality: Never commits, flushes, or rolls back implicitly.
    - Exception Wrapping: Translates database driver exceptions into domain PersistenceError.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the User repository.

        Args:
            session: Active SQLAlchemy asynchronous database session.
        """
        self._session = session

    async def get_by_id(
        self,
        user_id: uuid.UUID,
    ) -> Optional[User]:
        """
        Retrieve a user record by primary key UUID.

        Args:
            user_id: UUID primary key.

        Returns:
            The User instance if found, otherwise None.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = select(User).where(User.id == user_id)
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve user by ID '{user_id}': {exc}"
            ) from exc

    async def get_by_username(
        self,
        username: str,
    ) -> Optional[User]:
        """
        Retrieve a user record by username (case-sensitive exact match).

        Args:
            username: Target username string.

        Returns:
            The User instance if found, otherwise None.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = select(User).where(User.username == username.strip())
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve user by username '{username}': {exc}"
            ) from exc

    async def get_by_email(
        self,
        email: str,
    ) -> Optional[User]:
        """
        Retrieve a user record by email address (case-insensitive lookup).

        Args:
            email: Target email address string.

        Returns:
            The User instance if found, otherwise None.

        Raises:
            PersistenceError: On database query failure.
        """
        try:
            stmt = select(User).where(func.lower(User.email) == email.strip().lower())
            result = await self._session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to retrieve user by email '{email}': {exc}"
            ) from exc

    async def list_users(
        self,
        limit: int = 50,
        offset: int = 0,
        is_active: Optional[bool] = None,
        role: Optional[Union[UserRole, AuditActorType, str]] = None,
    ) -> Tuple[List[User], int]:
        """
        Retrieve paginated list of users with optional status and role filtering.

        Args:
            limit: Maximum items per page (default: 50).
            offset: Items to skip (default: 0).
            is_active: Optional boolean filter for active status.
            role: Optional AuditActorType or role string filter.

        Returns:
            Tuple of (list of User records, total matching count).

        Raises:
            ValueError: If limit <= 0 or offset < 0.
            PersistenceError: On database query failure.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if offset < 0:
            raise ValueError(f"offset cannot be negative, got {offset}")

        try:
            base_stmt = select(User)
            count_stmt = select(func.count(User.id))

            filters = []
            if is_active is not None:
                filters.append(User.is_active == is_active)

            if role is not None:
                role_str = role.value if hasattr(role, "value") else str(role).strip().upper()
                try:
                    role_val = UserRole(role_str)
                except ValueError:
                    role_val = role_str
                filters.append(User.role == role_val)

            if filters:
                base_stmt = base_stmt.where(*filters)
                count_stmt = count_stmt.where(*filters)

            # Deterministic sorting
            base_stmt = (
                base_stmt.order_by(User.created_at.desc(), User.id.desc())
                .limit(limit)
                .offset(offset)
            )

            total_res = await self._session.execute(count_stmt)
            total = total_res.scalar_one()

            users_res = await self._session.execute(base_stmt)
            users = list(users_res.scalars().all())

            return users, total
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Failed to list users with filters is_active={is_active}, role={role}: {exc}"
            ) from exc

    async def add(
        self,
        user: User,
    ) -> User:
        """
        Attach a new User instance to the session.

        Does not commit or flush; transaction boundary is caller-owned.

        Args:
            user: User entity to attach.

        Returns:
            The attached User entity.
        """
        self._session.add(user)
        return user

    async def add_many(
        self,
        users: Sequence[User],
    ) -> List[User]:
        """
        Attach multiple User instances to the session in batch.

        Args:
            users: Sequence of User entities.

        Returns:
            List of attached User entities preserving order.
        """
        if not users:
            return []
        user_list = list(users)
        self._session.add_all(user_list)
        return user_list

    async def update_last_login(
        self,
        user_id: uuid.UUID,
        login_time: Optional[datetime] = None,
    ) -> Optional[User]:
        """
        Update the last_login_at timestamp for a specified user.

        Args:
            user_id: Target user UUID.
            login_time: Optional timestamp (defaults to current UTC time).

        Returns:
            The updated User entity if found, otherwise None.

        Raises:
            PersistenceError: On database query failure.
        """
        user = await self.get_by_id(user_id)
        if not user:
            return None
        user.last_login_at = login_time or datetime.now(timezone.utc)
        return user


__all__ = ["UserRepository"]
