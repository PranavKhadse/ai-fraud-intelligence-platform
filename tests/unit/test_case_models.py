"""
Unit Tests for Phase 12.1: Case & CaseNote SQLAlchemy 2.0 ORM Models.

Validates the declarative schema, table metadata, columns, types, foreign keys,
uniqueness constraints, check constraints, indexes, and relationships for
the Case and CaseNote entities without requiring a live PostgreSQL connection.
"""

from datetime import datetime, timezone
import uuid
import pytest
from sqlalchemy import inspect, Uuid, DateTime, String, Text, Enum
from sqlalchemy.orm import RelationshipDirection

from backend.app.db.models import (
    Base,
    Transaction,
    RiskEvaluation,
    AuditLog,
    Case,
    CaseNote,
    CaseStatus,
    CasePriority,
    CaseDisposition,
    CaseTriggerSource,
    CaseNoteType,
    AuditActorType,
    AuditEntityType,
)


class TestDeclarativeCaseModels:
    """Validates Case and CaseNote model registration and metadata structure."""

    def test_case_tables_registered_in_metadata(self):
        """Ensure both cases and case_notes tables are registered in Base.metadata."""
        tables = Base.metadata.tables
        assert "cases" in tables
        assert "case_notes" in tables

    def test_case_primary_key(self):
        """Verify Case primary key is a non-nullable UUID."""
        table = Case.__table__
        assert len(table.primary_key.columns) == 1
        pk = table.primary_key.columns["id"]
        assert isinstance(pk.type, Uuid)
        assert pk.nullable is False

    def test_case_note_primary_key(self):
        """Verify CaseNote primary key is a non-nullable UUID."""
        table = CaseNote.__table__
        assert len(table.primary_key.columns) == 1
        pk = table.primary_key.columns["id"]
        assert isinstance(pk.type, Uuid)
        assert pk.nullable is False


class TestCaseTableColumnsAndConstraints:
    """Validates Column types, nullability, uniqueness, and check constraints on cases table."""

    def test_case_number_unique_and_indexed(self):
        table = Case.__table__
        col = table.columns["case_number"]
        assert isinstance(col.type, String)
        assert col.type.length == 32
        assert col.nullable is False
        assert col.unique is True or any(
            idx.unique and "case_number" in [c.name for c in idx.columns]
            for idx in table.indexes
        )

    def test_transaction_id_unique_foreign_key(self):
        table = Case.__table__
        col = table.columns["transaction_id"]
        assert isinstance(col.type, Uuid)
        assert col.nullable is False
        assert col.unique is True or any(
            idx.unique and "transaction_id" in [c.name for c in idx.columns]
            for idx in table.indexes
        )
        # Check foreign key to transactions.id with RESTRICT
        fk = list(col.foreign_keys)[0]
        assert fk.target_fullname == "transactions.id"
        assert fk.ondelete == "RESTRICT"

    def test_evaluation_id_foreign_key(self):
        table = Case.__table__
        col = table.columns["evaluation_id"]
        assert isinstance(col.type, Uuid)
        assert col.nullable is False
        fk = list(col.foreign_keys)[0]
        assert fk.target_fullname == "risk_evaluations.id"
        assert fk.ondelete == "RESTRICT"

    def test_case_enums(self):
        table = Case.__table__

        # Status
        status_col = table.columns["status"]
        assert isinstance(status_col.type, Enum)
        assert set(status_col.type.enums) == {"OPEN", "IN_REVIEW", "ESCALATED", "RESOLVED", "CLOSED"}

        # Priority
        prio_col = table.columns["priority"]
        assert isinstance(prio_col.type, Enum)
        assert set(prio_col.type.enums) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

        # Trigger Source
        trigger_col = table.columns["trigger_source"]
        assert isinstance(trigger_col.type, Enum)
        assert set(trigger_col.type.enums) == {
            "AUTOMATED_REVIEW_POLICY",
            "AUTOMATED_RULE_OVERRIDE",
            "MANUAL_ANALYST_ESCALATION",
        }

        # Disposition
        disp_col = table.columns["disposition"]
        assert isinstance(disp_col.type, Enum)
        assert set(disp_col.type.enums) == {
            "CONFIRMED_FRAUD",
            "FALSE_POSITIVE",
            "LEGITIMATE",
            "SUSPICIOUS_RESOLVED",
        }
        assert disp_col.nullable is True

    def test_case_check_constraints(self):
        table = Case.__table__
        constraint_names = {c.name for c in table.constraints if hasattr(c, "name")}
        assert "chk_cases_disposition_state" in constraint_names

    def test_case_indexes(self):
        table = Case.__table__
        index_names = {idx.name for idx in table.indexes}
        assert "ix_cases_status_priority_created" in index_names
        assert "ix_cases_assigned_status" in index_names


class TestCaseNoteTableColumnsAndConstraints:
    """Validates Column types, nullability, and constraints on case_notes table."""

    def test_case_note_foreign_key_cascade(self):
        table = CaseNote.__table__
        col = table.columns["case_id"]
        assert isinstance(col.type, Uuid)
        assert col.nullable is False
        fk = list(col.foreign_keys)[0]
        assert fk.target_fullname == "cases.id"
        assert fk.ondelete == "CASCADE"

    def test_case_note_author_and_type(self):
        table = CaseNote.__table__

        author_col = table.columns["author_id"]
        assert isinstance(author_col.type, String)
        assert author_col.type.length == 128
        assert author_col.nullable is False

        role_col = table.columns["author_role"]
        assert isinstance(role_col.type, Enum)
        assert set(role_col.type.enums) == {"SYSTEM", "ANALYST", "ADMIN", "API_CLIENT"}

        type_col = table.columns["note_type"]
        assert isinstance(type_col.type, Enum)
        assert set(type_col.type.enums) == {
            "INVESTIGATION",
            "ESCALATION",
            "DISPOSITION",
            "SYSTEM_AUDIT",
        }

    def test_case_note_content_non_empty_check(self):
        table = CaseNote.__table__
        constraint_names = {c.name for c in table.constraints if hasattr(c, "name")}
        assert "chk_case_notes_content_non_empty" in constraint_names


class TestCaseRelationships:
    """Validates ORM relationship definitions between Case, Transaction, RiskEvaluation, and CaseNote."""

    def test_case_to_transaction_relationship(self):
        mapper = inspect(Case)
        rel = mapper.relationships["transaction"]
        assert rel.direction == RelationshipDirection.MANYTOONE
        assert rel.target.name == "transactions"

        # Reverse relationship on Transaction
        tx_mapper = inspect(Transaction)
        assert "case" in tx_mapper.relationships
        assert tx_mapper.relationships["case"].uselist is False

    def test_case_to_evaluation_relationship(self):
        mapper = inspect(Case)
        rel = mapper.relationships["evaluation"]
        assert rel.direction == RelationshipDirection.MANYTOONE
        assert rel.target.name == "risk_evaluations"

        # Reverse relationship on RiskEvaluation
        eval_mapper = inspect(RiskEvaluation)
        assert "case" in eval_mapper.relationships
        assert eval_mapper.relationships["case"].uselist is False

    def test_case_to_notes_relationship_cascade(self):
        mapper = inspect(Case)
        rel = mapper.relationships["notes"]
        assert rel.direction == RelationshipDirection.ONETOMANY
        assert rel.target.name == "case_notes"
        assert rel.cascade.delete is True
        assert rel.cascade.delete_orphan is True


class TestCaseEntityInstantiation:
    """Validates transient instantiation, defaults, and repr strings for Case and CaseNote."""

    def test_case_instantiation(self):
        case_id = uuid.uuid4()
        tx_id = uuid.uuid4()
        eval_id = uuid.uuid4()

        case = Case(
            id=case_id,
            case_number="CASE-20260918-A1B2C3",
            transaction_id=tx_id,
            evaluation_id=eval_id,
            status=CaseStatus.OPEN,
            priority=CasePriority.MEDIUM,
            trigger_source=CaseTriggerSource.AUTOMATED_REVIEW_POLICY,
        )

        assert case.id == case_id
        assert case.case_number == "CASE-20260918-A1B2C3"
        assert case.status == CaseStatus.OPEN
        assert case.priority == CasePriority.MEDIUM
        assert case.trigger_source == CaseTriggerSource.AUTOMATED_REVIEW_POLICY
        assert case.assigned_to is None
        assert case.disposition is None
        assert "CASE-20260918-A1B2C3" in repr(case)

    def test_case_note_instantiation(self):
        note_id = uuid.uuid4()
        case_id = uuid.uuid4()

        note = CaseNote(
            id=note_id,
            case_id=case_id,
            author_id="analyst_sarah",
            author_role=AuditActorType.ANALYST,
            note_type=CaseNoteType.INVESTIGATION,
            content="Cardholder contacted via phone; confirmed recent travel.",
        )

        assert note.id == note_id
        assert note.case_id == case_id
        assert note.author_id == "analyst_sarah"
        assert note.author_role == AuditActorType.ANALYST
        assert note.note_type == CaseNoteType.INVESTIGATION
        assert "Cardholder contacted" in note.content
        assert "analyst_sarah" in repr(note)

    def test_audit_entity_type_case_enum_value(self):
        assert AuditEntityType.CASE == "CASE"
        assert AuditEntityType.CASE.value == "CASE"
