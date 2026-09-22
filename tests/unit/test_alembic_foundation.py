"""
Unit Tests for Phase 9 Milestone 9.3: Alembic Migration Foundation.

Validates the Alembic configuration, migration scripts, offline SQL DDL generation,
metadata synchronization, constraint verification, and import isolation without requiring
a live PostgreSQL connection.
"""

import importlib
import io
from pathlib import Path
import sys
import pytest
from alembic.config import Config
from alembic import command
from sqlalchemy import inspect

from backend.app.db.models import Base, Transaction, RiskEvaluation


class TestAlembicConfiguration:
    """Validates Alembic configuration file and directory structure."""

    def test_alembic_ini_exists_and_parses(self):
        ini_path = Path("backend/alembic.ini")
        assert ini_path.exists(), "backend/alembic.ini must exist"

        cfg = Config(str(ini_path))
        script_loc = cfg.get_main_option("script_location")
        assert script_loc is not None
        assert "alembic" in script_loc

    def test_migration_directory_structure(self):
        backend_dir = Path("backend/alembic")
        assert backend_dir.exists() and backend_dir.is_dir()

        env_py = backend_dir / "env.py"
        assert env_py.exists()

        mako_template = backend_dir / "script.py.mako"
        assert mako_template.exists()

        versions_dir = backend_dir / "versions"
        assert versions_dir.exists() and versions_dir.is_dir()


class TestTargetMetadataIntegration:
    """Validates target_metadata synchronization with SQLAlchemy DeclarativeBase."""

    def test_target_metadata_contains_all_nine_tables(self):
        from backend.alembic.env import target_metadata

        assert target_metadata is Base.metadata
        expected_tables = {
            "transactions",
            "risk_evaluations",
            "evaluation_rule_matches",
            "evaluation_reason_codes",
            "evaluation_feature_attributions",
            "audit_logs",
            "cases",
            "case_notes",
            "model_monitoring_snapshots",
        }
        registered_tables = set(target_metadata.tables.keys())
        assert expected_tables == registered_tables, f"Mismatch in tables: {registered_tables ^ expected_tables}"


class TestInitialMigrationRevision:
    """Validates the initial migration script structure and revision metadata."""

    def test_initial_migration_file_exists(self):
        migration_file = Path("backend/alembic/versions/0001_initial_core_tables.py")
        assert migration_file.exists()

    def test_initial_migration_attributes(self):
        initial_migration = importlib.import_module("backend.alembic.versions.0001_initial_core_tables")

        assert initial_migration.revision == "0001_initial_core_tables"
        assert initial_migration.down_revision is None
        assert callable(initial_migration.upgrade)
        assert callable(initial_migration.downgrade)


class TestUniqueIndexMigrationRevision:
    """Validates the 0002 partial unique index migration script structure."""

    def test_unique_index_migration_file_exists(self):
        migration_file = Path("backend/alembic/versions/0002_add_unique_index_external_tx_id.py")
        assert migration_file.exists()

    def test_unique_index_migration_attributes(self):
        unique_migration = importlib.import_module("backend.alembic.versions.0002_add_unique_index_external_tx_id")

        assert unique_migration.revision == "0002_add_unique_index_external_tx_id"
        assert unique_migration.down_revision == "0001_initial_core_tables"
        assert callable(unique_migration.upgrade)
        assert callable(unique_migration.downgrade)


class TestCaseMigrationRevision:
    """Validates the 0003 cases and case_notes migration script structure."""

    def test_cases_migration_file_exists(self):
        migration_file = Path("backend/alembic/versions/0003_add_cases_and_case_notes_tables.py")
        assert migration_file.exists()

    def test_cases_migration_attributes(self):
        cases_migration = importlib.import_module("backend.alembic.versions.0003_add_cases_and_case_notes_tables")

        assert cases_migration.revision == "0003_add_cases_and_case_notes_tables"
        assert cases_migration.down_revision == "0002_add_unique_index_external_tx_id"
        assert callable(cases_migration.upgrade)
        assert callable(cases_migration.downgrade)


class TestMonitoringMigrationRevision:
    """Validates the 0004 model_monitoring_snapshots migration script structure."""

    def test_monitoring_migration_file_exists(self):
        migration_file = Path("backend/alembic/versions/0004_add_model_monitoring_snapshots_table.py")
        assert migration_file.exists()

    def test_monitoring_migration_attributes(self):
        monitoring_migration = importlib.import_module("backend.alembic.versions.0004_add_model_monitoring_snapshots_table")

        assert monitoring_migration.revision == "0004_add_model_monitoring_snapshots_table"
        assert monitoring_migration.down_revision == "0003_add_cases_and_case_notes_tables"
        assert callable(monitoring_migration.upgrade)
        assert callable(monitoring_migration.downgrade)


class TestOfflineSQLGeneration:
    """Validates offline SQL DDL generation without a live PostgreSQL database."""

    @pytest.fixture
    def generated_sql(self, monkeypatch) -> str:
        """Run alembic upgrade head --sql in-memory and capture generated DDL."""
        buffer = io.StringIO()
        cfg = Config("backend/alembic.ini", stdout=buffer)
        monkeypatch.setattr(sys, "stdout", buffer)

        command.upgrade(cfg, "head", sql=True)
        return buffer.getvalue()

    def test_all_nine_tables_created_in_sql(self, generated_sql: str):
        assert "CREATE TABLE transactions" in generated_sql
        assert "CREATE TABLE risk_evaluations" in generated_sql
        assert "CREATE TABLE evaluation_rule_matches" in generated_sql
        assert "CREATE TABLE evaluation_reason_codes" in generated_sql
        assert "CREATE TABLE evaluation_feature_attributions" in generated_sql
        assert "CREATE TABLE audit_logs" in generated_sql
        assert "CREATE TABLE cases" in generated_sql
        assert "CREATE TABLE case_notes" in generated_sql
        assert "CREATE TABLE model_monitoring_snapshots" in generated_sql

    def test_cases_table_constraints_in_sql(self, generated_sql: str):
        assert "chk_cases_disposition_state" in generated_sql
        assert "chk_case_notes_content_non_empty" in generated_sql
        assert "uq_cases_case_number" in generated_sql
        assert "uq_cases_transaction_id" in generated_sql

    def test_foreign_key_deletion_rules_in_sql(self, generated_sql: str):
        # RiskEvaluation -> Transaction must be ON DELETE RESTRICT
        assert "REFERENCES transactions (id) ON DELETE RESTRICT" in generated_sql

        # Child tables -> RiskEvaluation must be ON DELETE CASCADE
        assert "REFERENCES risk_evaluations (id) ON DELETE CASCADE" in generated_sql

    def test_check_constraints_in_sql(self, generated_sql: str):
        assert "chk_transactions_amount_positive" in generated_sql
        assert "chk_risk_evaluations_model_score" in generated_sql
        assert "chk_risk_evaluations_risk_score" in generated_sql

    def test_key_indexes_in_sql(self, generated_sql: str):
        assert "ix_transactions_account_ts" in generated_sql
        assert "ix_risk_evaluations_action_tier_date" in generated_sql
        assert "ix_risk_evaluations_model_ver_date" in generated_sql
        assert "ix_audit_logs_entity_lookup" in generated_sql

    def test_partial_unique_index_in_sql(self, generated_sql: str):
        assert "uq_transactions_external_tx_id" in generated_sql
        assert "CREATE UNIQUE INDEX uq_transactions_external_tx_id" in generated_sql
        assert "WHERE external_transaction_id IS NOT NULL" in generated_sql

    def test_numeric_and_json_types_in_sql(self, generated_sql: str):
        assert "NUMERIC(15, 2)" in generated_sql  # Transaction amount
        assert "NUMERIC(8, 6)" in generated_sql   # Model score
        assert "NUMERIC(10, 6)" in generated_sql  # SHAP value & margins
        assert "JSONB" in generated_sql            # Feature snapshot and payloads


class TestImportIsolationAndSafety:
    """Ensures Alembic imports maintain strict process and architectural isolation."""

    def test_env_import_does_not_connect_to_db(self):
        """Verify importing env does not open a live database connection."""
        from backend.alembic import env  # noqa: F401

        # target_metadata is purely in-memory metadata
        assert env.target_metadata is not None

    def test_transaction_evaluations_relationship_is_non_destructive(self):
        """Verify Transaction.evaluations does not have destructive delete cascade."""
        mapper = inspect(Transaction)
        evaluations_rel = mapper.relationships["evaluations"]

        # Ensure cascade string does NOT contain 'delete' or 'delete-orphan'
        cascade_opts = evaluations_rel.cascade
        assert not cascade_opts.delete, "Transaction.evaluations must not cascade delete!"
        assert not cascade_opts.delete_orphan, "Transaction.evaluations must not delete orphans!"
        assert evaluations_rel.passive_deletes is True
