"""
Integration Tests for Phase 15.1: Alembic Migration 0006 Upgrade/Downgrade Lifecycle.

Validates:
- Successful live execution of Alembic migration 0006 against real PostgreSQL test database.
- Column presence: prev_hash and entry_hash in audit_logs.
- Table presence and constraints: users and api_keys tables.
- Verification of extended AuditEntityType enums (USER, API_KEY, MODEL) in PostgreSQL.
- Deterministic downgrade to 0005: drops users and api_keys, removes hash columns.
- Re-upgrade to head (0006).
"""

import asyncio
import os
import uuid
from alembic import command
from alembic.config import Config
import pytest
import pytest_asyncio
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from backend.app.db.models.audit_log import AuditLog
from backend.app.db.models.enums import AuditActorType, AuditEntityType
from backend.app.db.models.user import User
from backend.app.db.models.api_key import APIKey
from tests.conftest import _check_postgres_connection, get_test_database_url


@pytest.fixture(scope="module")
def postgres_test_url() -> str:
    """Retrieve test PostgreSQL connection URL."""
    return get_test_database_url()


@pytest.fixture(scope="module")
def is_postgres_available(postgres_test_url: str) -> bool:
    """Check if PostgreSQL is reachable."""
    return asyncio.run(_check_postgres_connection(postgres_test_url))


@pytest.mark.asyncio
async def test_migration_0006_upgrade_downgrade_cycle(postgres_test_url: str, is_postgres_available: bool):
    """
    Test live migration 0006 upgrade, schema verification, entity insertion,
    downgrade to 0005, and re-upgrade back to head.
    """
    if not is_postgres_available:
        pytest.skip("PostgreSQL test database not available. Skipping live migration test.")

    cfg = Config("backend/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", postgres_test_url)

    engine = create_async_engine(postgres_test_url, poolclass=NullPool)

    async with engine.connect() as conn:
        has_version_tbl = await conn.scalar(
            text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'alembic_version')")
        )
        if not has_version_tbl:
            command.stamp(cfg, "0005_add_model_registry_table")
        else:
            current_ver = await conn.scalar(text("SELECT version_num FROM alembic_version LIMIT 1"))
            if current_ver == "0006_auth_and_audit_hash_chain":
                command.downgrade(cfg, "0005_add_model_registry_table")

    # 1. Upgrade to head (applies migration 0006)
    command.upgrade(cfg, "head")

    try:
        # 2. Verify schema columns and tables in live PostgreSQL
        async with engine.connect() as conn:
            # Check users table exists
            users_exist = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users')")
            )
            assert users_exist is True

            # Check api_keys table exists
            api_keys_exist = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'api_keys')")
            )
            assert api_keys_exist is True

            # Check audit_logs columns prev_hash and entry_hash
            prev_hash_col = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'audit_logs' AND column_name = 'prev_hash')")
            )
            assert prev_hash_col is True

            entry_hash_col = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'audit_logs' AND column_name = 'entry_hash')")
            )
            assert entry_hash_col is True

            # 3. Test insert into users table
            user_id = uuid.uuid4()
            await conn.execute(
                text(
                    """
                    INSERT INTO users (id, username, email, hashed_password, role, is_active, is_superuser, created_at, updated_at)
                    VALUES (:id, :username, :email, :hashed_password, :role, :is_active, :is_superuser, NOW(), NOW())
                    """
                ),
                {
                    "id": user_id,
                    "username": f"test_analyst_{uuid.uuid4().hex[:6]}",
                    "email": f"analyst_{uuid.uuid4().hex[:6]}@example.com",
                    "hashed_password": "hashed_bcrypt_string",
                    "role": "ANALYST",
                    "is_active": True,
                    "is_superuser": False,
                },
            )

            # 3b. Verify chk_users_role_human: 'SYSTEM' and 'API_CLIENT' are rejected
            with pytest.raises(IntegrityError):
                async with conn.begin_nested():
                    await conn.execute(
                        text(
                            """
                            INSERT INTO users (id, username, email, hashed_password, role, is_active, is_superuser, created_at, updated_at)
                            VALUES (:id, :username, :email, :hashed_password, 'SYSTEM', true, false, NOW(), NOW())
                            """
                        ),
                        {
                            "id": uuid.uuid4(),
                            "username": f"sys_spoof_{uuid.uuid4().hex[:6]}",
                            "email": f"sys_{uuid.uuid4().hex[:6]}@example.com",
                            "hashed_password": "hashed_bcrypt_string",
                        },
                    )

            with pytest.raises(IntegrityError):
                async with conn.begin_nested():
                    await conn.execute(
                        text(
                            """
                            INSERT INTO users (id, username, email, hashed_password, role, is_active, is_superuser, created_at, updated_at)
                            VALUES (:id, :username, :email, :hashed_password, 'API_CLIENT', true, false, NOW(), NOW())
                            """
                        ),
                        {
                            "id": uuid.uuid4(),
                            "username": f"cli_spoof_{uuid.uuid4().hex[:6]}",
                            "email": f"cli_{uuid.uuid4().hex[:6]}@example.com",
                            "hashed_password": "hashed_bcrypt_string",
                        },
                    )

            # 3c. Verify valid human role 'ADMIN' succeeds
            admin_user_id = uuid.uuid4()
            await conn.execute(
                text(
                    """
                    INSERT INTO users (id, username, email, hashed_password, role, is_active, is_superuser, created_at, updated_at)
                    VALUES (:id, :username, :email, :hashed_password, 'ADMIN', true, true, NOW(), NOW())
                    """
                ),
                {
                    "id": admin_user_id,
                    "username": f"test_admin_{uuid.uuid4().hex[:6]}",
                    "email": f"admin_{uuid.uuid4().hex[:6]}@example.com",
                    "hashed_password": "hashed_bcrypt_string",
                },
            )

            # 4. Test insert into api_keys table
            key_id = uuid.uuid4()
            await conn.execute(
                text(
                    """
                    INSERT INTO api_keys (id, key_prefix, hashed_key, name, client_id, role, is_active, user_id, created_at, updated_at)
                    VALUES (:id, :key_prefix, :hashed_key, :name, :client_id, :role, :is_active, :user_id, NOW(), NOW())
                    """
                ),
                {
                    "id": key_id,
                    "key_prefix": "pk_live_",
                    "hashed_key": f"hash_{uuid.uuid4().hex}",
                    "name": "Integration Key",
                    "client_id": "test_client",
                    "role": "API_CLIENT",
                    "is_active": True,
                    "user_id": user_id,
                },
            )

            # 4b. Verify chk_api_keys_role_machine: 'ADMIN', 'ANALYST', 'SYSTEM' are rejected
            for forbidden_key_role in ("ADMIN", "ANALYST", "SYSTEM"):
                with pytest.raises(IntegrityError):
                    async with conn.begin_nested():
                        await conn.execute(
                            text(
                                """
                                INSERT INTO api_keys (id, key_prefix, hashed_key, name, client_id, role, is_active, user_id, created_at, updated_at)
                                VALUES (:id, :key_prefix, :hashed_key, :name, :client_id, :role, :is_active, :user_id, NOW(), NOW())
                                """
                            ),
                            {
                                "id": uuid.uuid4(),
                                "key_prefix": "pk_live_",
                                "hashed_key": f"hash_bad_{uuid.uuid4().hex}",
                                "name": "Illegal Role Key",
                                "client_id": "test_client",
                                "role": forbidden_key_role,
                                "is_active": True,
                                "user_id": user_id,
                            },
                        )

            # 5. Test insert into audit_logs with hash chain and extended entity types
            log_id = uuid.uuid4()
            await conn.execute(
                text(
                    """
                    INSERT INTO audit_logs (id, event_type, entity_type, entity_id, action, actor_type, actor_id, prev_hash, entry_hash, event_timestamp, created_at)
                    VALUES (:id, :event_type, :entity_type, :entity_id, :action, :actor_type, :actor_id, :prev_hash, :entry_hash, NOW(), NOW())
                    """
                ),
                {
                    "id": log_id,
                    "event_type": "USER_CREATED",
                    "entity_type": "USER",
                    "entity_id": user_id,
                    "action": "CREATE",
                    "actor_type": "ADMIN",
                    "actor_id": "admin_root",
                    "prev_hash": "0" * 64,
                    "entry_hash": "a" * 64,
                },
            )

            # Also test entity_type = MODEL and API_KEY
            for etype in ("MODEL", "API_KEY"):
                await conn.execute(
                    text(
                        """
                        INSERT INTO audit_logs (id, event_type, entity_type, entity_id, action, actor_type, actor_id, prev_hash, entry_hash, event_timestamp, created_at)
                        VALUES (:id, :event_type, :entity_type, :entity_id, :action, :actor_type, :actor_id, :prev_hash, :entry_hash, NOW(), NOW())
                        """
                    ),
                    {
                        "id": uuid.uuid4(),
                        "event_type": f"{etype}_EVENT",
                        "entity_type": etype,
                        "entity_id": uuid.uuid4(),
                        "action": "ACTION",
                        "actor_type": "SYSTEM",
                        "actor_id": "system_daemon",
                        "prev_hash": "a" * 64,
                        "entry_hash": "b" * 64,
                    },
                )

            await conn.commit()

            # Clean test rows to allow clean downgrade
            await conn.execute(text("DELETE FROM audit_logs WHERE entity_type IN ('USER', 'API_KEY', 'MODEL')"))
            await conn.execute(text("DELETE FROM api_keys WHERE id = :id"), {"id": key_id})
            await conn.execute(text("DELETE FROM users WHERE id IN (:id1, :id2)"), {"id1": user_id, "id2": admin_user_id})
            await conn.commit()

        # 6. Test downgrade to 0005
        command.downgrade(cfg, "0005_add_model_registry_table")

        async with engine.connect() as conn:
            # Verify users and api_keys tables are dropped
            users_exist_after = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users')")
            )
            assert users_exist_after is False

            api_keys_exist_after = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'api_keys')")
            )
            assert api_keys_exist_after is False

            # Verify prev_hash and entry_hash are dropped from audit_logs
            prev_hash_col_after = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'audit_logs' AND column_name = 'prev_hash')")
            )
            assert prev_hash_col_after is False

            entry_hash_col_after = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'audit_logs' AND column_name = 'entry_hash')")
            )
            assert entry_hash_col_after is False

        # 7. Re-upgrade back to head for subsequent tests
        command.upgrade(cfg, "head")

        async with engine.connect() as conn:
            re_upgraded = await conn.scalar(
                text("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users')")
            )
            assert re_upgraded is True

    finally:
        await engine.dispose()
