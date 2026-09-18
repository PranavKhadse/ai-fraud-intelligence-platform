"""
Unit Tests for Phase 9 Milestone 9.2: SQLAlchemy 2.0 ORM Models.

Validates the declarative schema, table metadata, columns, types, foreign keys,
indices, check constraints, relationship configurations, and enum persistence
without requiring a live PostgreSQL connection.
"""

from datetime import datetime, timezone
from decimal import Decimal
import uuid
import pytest
from sqlalchemy import inspect, Numeric, Uuid, DateTime, String, SmallInteger, Integer, Boolean, Text
from sqlalchemy.orm import RelationshipDirection

from backend.app.db.models import (
    Base,
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Transaction,
    RiskEvaluation,
    EvaluationRuleMatch,
    EvaluationReasonCode,
    EvaluationFeatureAttribution,
    AuditLog,
    DecisionAction,
    RiskTier,
    PolicyMode,
    RuleOutcome,
    RuleType,
    AttributionDirection,
    ReasonSource,
    ReasonSeverity,
    AuditActorType,
    AuditEntityType,
)


class TestDeclarativeMetadata:
    """Validates the SQLAlchemy DeclarativeBase and registered table schemas."""

    def test_all_models_registered_in_metadata(self):
        """Ensure all required models are in the declarative metadata."""
        expected_tables = {
            "transactions",
            "risk_evaluations",
            "evaluation_rule_matches",
            "evaluation_reason_codes",
            "evaluation_feature_attributions",
            "audit_logs",
            "cases",
            "case_notes",
        }
        registered_tables = set(Base.metadata.tables.keys())
        assert expected_tables.issubset(registered_tables)

    def test_case_models_defined_in_package(self):
        """Explicitly verify that Case and CaseNote models and enums are present in models package."""
        import backend.app.db.models as models_pkg

        assert hasattr(models_pkg, "Case")
        assert hasattr(models_pkg, "CaseNote")
        assert hasattr(models_pkg, "CaseStatus")
        assert hasattr(models_pkg, "CasePriority")
        assert hasattr(models_pkg, "CaseDisposition")
        assert hasattr(models_pkg, "CaseTriggerSource")
        assert hasattr(models_pkg, "CaseNoteType")


class TestTransactionModel:
    """Validates the Transaction ORM model schema and constraints."""

    def test_table_name_and_primary_key(self):
        table = Transaction.__table__
        assert table.name == "transactions"
        assert len(table.primary_key.columns) == 1
        pk = table.primary_key.columns["id"]
        assert isinstance(pk.type, Uuid)
        assert pk.nullable is False

    def test_amount_precision(self):
        table = Transaction.__table__
        amount_col = table.columns["amount"]
        assert isinstance(amount_col.type, Numeric)
        assert amount_col.type.precision == 15
        assert amount_col.type.scale == 2
        assert amount_col.nullable is False

    def test_coordinate_precision(self):
        table = Transaction.__table__
        for col_name in ["cardholder_lat", "cardholder_long", "merchant_lat", "merchant_long"]:
            col = table.columns[col_name]
            assert isinstance(col.type, Numeric)
            assert col.type.precision == 9
            assert col.type.scale == 6
            assert col.nullable is False

    def test_features_snapshot_column(self):
        table = Transaction.__table__
        col = table.columns["features_snapshot"]
        assert col.nullable is False

    def test_external_transaction_id_not_globally_unique(self):
        table = Transaction.__table__
        col = table.columns["external_transaction_id"]
        assert col.unique is not True or col.unique is None

    def test_indexes_and_check_constraints(self):
        table = Transaction.__table__
        check_constraints = [c.name for c in table.constraints if hasattr(c, "name") and c.name]
        assert "chk_transactions_amount_positive" in check_constraints

        index_names = [idx.name for idx in table.indexes]
        assert "ix_transactions_account_ts" in index_names

    def test_instantiation_and_repr_masking(self):
        tx = Transaction(
            account_id="acc_12345",
            merchant_category="grocery_pos",
            job_category="engineer",
            amount=Decimal("125.50"),
            currency="USD",
            cardholder_lat=Decimal("40.712800"),
            cardholder_long=Decimal("-74.006000"),
            merchant_lat=Decimal("40.713000"),
            merchant_long=Decimal("-74.005800"),
            city_pop=500000,
            transaction_timestamp=datetime.now(timezone.utc),
            features_snapshot={"amt": 125.50, "city_pop": 500000},
        )
        rep = repr(tx)
        assert "Transaction" in rep
        assert "acc_12345" in rep
        assert "125.50" in rep
        assert "features_snapshot" not in rep  # Does not dump large/sensitive feature dict


class TestRiskEvaluationModel:
    """Validates the RiskEvaluation ORM model schema and relationships."""

    def test_table_name_and_foreign_keys(self):
        table = RiskEvaluation.__table__
        assert table.name == "risk_evaluations"
        assert len(table.foreign_keys) == 1
        fk = list(table.foreign_keys)[0]
        assert fk.column.table.name == "transactions"
        assert fk.column.name == "id"
        assert fk.ondelete == "RESTRICT"

    def test_score_and_tier_columns(self):
        table = RiskEvaluation.__table__
        model_score_col = table.columns["model_score"]
        assert isinstance(model_score_col.type, Numeric)
        assert model_score_col.type.precision == 8
        assert model_score_col.type.scale == 6

        risk_score_col = table.columns["risk_score"]
        assert isinstance(risk_score_col.type, SmallInteger)

    def test_margin_and_latency_columns(self):
        table = RiskEvaluation.__table__
        for col_name in ["output_margin", "base_value"]:
            col = table.columns[col_name]
            assert isinstance(col.type, Numeric)
            assert col.type.precision == 10
            assert col.type.scale == 6
            assert col.nullable is True

        lat_col = table.columns["evaluation_latency_ms"]
        assert isinstance(lat_col.type, Numeric)
        assert lat_col.type.precision == 8
        assert lat_col.type.scale == 2
        assert lat_col.nullable is True

    def test_check_constraints(self):
        table = RiskEvaluation.__table__
        constraint_names = [c.name for c in table.constraints if hasattr(c, "name") and c.name]
        assert "chk_risk_evaluations_model_score" in constraint_names
        assert "chk_risk_evaluations_risk_score" in constraint_names

    def test_relationships_defined(self):
        mapper = inspect(RiskEvaluation)
        relationships = {rel.key: rel for rel in mapper.relationships}
        assert "transaction" in relationships
        assert relationships["transaction"].direction == RelationshipDirection.MANYTOONE
        assert "rule_matches" in relationships
        assert relationships["rule_matches"].direction == RelationshipDirection.ONETOMANY
        assert "reason_codes" in relationships
        assert relationships["reason_codes"].direction == RelationshipDirection.ONETOMANY
        assert "feature_attributions" in relationships
        assert relationships["feature_attributions"].direction == RelationshipDirection.ONETOMANY

    def test_instantiation_and_repr(self):
        tx_id = uuid.uuid4()
        eval_record = RiskEvaluation(
            transaction_id=tx_id,
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            model_score=Decimal("0.854321"),
            risk_score=85,
            risk_tier=RiskTier.HIGH,
            decision_action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=False,
            decision_reason="High calibrated risk score.",
            evaluated_at=datetime.now(timezone.utc),
        )
        rep = repr(eval_record)
        assert "RiskEvaluation" in rep
        assert "HIGH" in rep
        assert "REVIEW" in rep
        assert "85" in rep


class TestChildModels:
    """Validates RuleMatch, ReasonCode, and FeatureAttribution models."""

    def test_rule_match_schema_and_fks(self):
        table = EvaluationRuleMatch.__table__
        assert table.name == "evaluation_rule_matches"
        assert len(table.foreign_keys) == 1
        fk = list(table.foreign_keys)[0]
        assert fk.column.table.name == "risk_evaluations"
        assert fk.column.name == "id"
        assert fk.ondelete == "CASCADE"
        assert isinstance(table.columns["priority"].type, SmallInteger)

    def test_reason_code_schema_and_fks(self):
        table = EvaluationReasonCode.__table__
        assert table.name == "evaluation_reason_codes"
        assert len(table.foreign_keys) == 1
        fk = list(table.foreign_keys)[0]
        assert fk.column.table.name == "risk_evaluations"
        assert fk.column.name == "id"
        assert fk.ondelete == "CASCADE"
        assert isinstance(table.columns["rank"].type, SmallInteger)

    def test_feature_attribution_schema_and_precision(self):
        table = EvaluationFeatureAttribution.__table__
        assert table.name == "evaluation_feature_attributions"
        assert len(table.foreign_keys) == 1
        fk = list(table.foreign_keys)[0]
        assert fk.column.table.name == "risk_evaluations"
        assert fk.column.name == "id"
        assert fk.ondelete == "CASCADE"

        shap_col = table.columns["shap_value"]
        assert isinstance(shap_col.type, Numeric)
        assert shap_col.type.precision == 10
        assert shap_col.type.scale == 6

        pct_col = table.columns["relative_contribution_pct"]
        assert isinstance(pct_col.type, Numeric)
        assert pct_col.type.precision == 6
        assert pct_col.type.scale == 4

    def test_child_repr_methods(self):
        eval_id = uuid.uuid4()
        rule = EvaluationRuleMatch(
            evaluation_id=eval_id,
            rule_id="RULE_VELOCITY_BURST",
            description="Rapid transaction count",
            feature_name="trans_count_1h",
            operator=">",
            comparison_value="5",
            outcome=RuleOutcome.REVIEW,
            rule_type=RuleType.VELOCITY,
            priority=10,
        )
        assert "RULE_VELOCITY_BURST" in repr(rule)

        reason = EvaluationReasonCode(
            evaluation_id=eval_id,
            code="VELOCITY_BURST_1H",
            headline="High transaction velocity",
            description="Transaction frequency spiked",
            category="VELOCITY",
            source=ReasonSource.MODEL,
            severity=ReasonSeverity.HIGH,
            rank=1,
        )
        assert "VELOCITY_BURST_1H" in repr(reason)

        attrib = EvaluationFeatureAttribution(
            evaluation_id=eval_id,
            feature_name="amt_to_avg_ratio_30d",
            display_name="Amount to 30d Average Ratio",
            raw_value=4.5,
            shap_value=Decimal("1.254300"),
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=Decimal("42.5000"),
            rank=1,
        )
        assert "amt_to_avg_ratio_30d" in repr(attrib)


class TestAuditLogModel:
    """Validates the AuditLog ORM model schema."""

    def test_audit_log_schema(self):
        table = AuditLog.__table__
        assert table.name == "audit_logs"
        assert len(table.foreign_keys) == 0  # Polymorphic / independent table

        assert isinstance(table.columns["id"].type, Uuid)
        assert isinstance(table.columns["event_type"].type, String)
        assert isinstance(table.columns["action"].type, String)

    def test_audit_log_indexes(self):
        table = AuditLog.__table__
        index_names = [idx.name for idx in table.indexes]
        assert "ix_audit_logs_entity_lookup" in index_names

    def test_audit_log_repr(self):
        log = AuditLog(
            event_type="PREDICTION_EXECUTED",
            entity_type=AuditEntityType.RISK_EVALUATION,
            entity_id=uuid.uuid4(),
            action="EVALUATE",
            actor_type=AuditActorType.API_CLIENT,
            actor_id="test_key_01",
            payload={"latency_ms": 12.4},
        )
        rep = repr(log)
        assert "AuditLog" in rep
        assert "PREDICTION_EXECUTED" in rep
        assert "API_CLIENT" in rep


class TestEnumPersistence:
    """Validates enum compatibility and stable string values."""

    def test_decision_action_values(self):
        assert DecisionAction.APPROVE.value == "APPROVE"
        assert DecisionAction.REVIEW.value == "REVIEW"
        assert DecisionAction.BLOCK.value == "BLOCK"

    def test_risk_tier_values(self):
        assert RiskTier.LOW.value == "LOW"
        assert RiskTier.MEDIUM.value == "MEDIUM"
        assert RiskTier.HIGH.value == "HIGH"
        assert RiskTier.CRITICAL.value == "CRITICAL"

    def test_policy_mode_values(self):
        assert PolicyMode.TRI_TIER.value == "TRI_TIER"
        assert PolicyMode.BINARY_AUTO.value == "BINARY_AUTO"

    def test_rule_outcome_values(self):
        assert RuleOutcome.BLOCK.value == "BLOCK"
        assert RuleOutcome.REVIEW.value == "REVIEW"
        assert RuleOutcome.MONITOR.value == "MONITOR"

    def test_reason_source_values(self):
        assert ReasonSource.MODEL.value == "MODEL"
        assert ReasonSource.RULE.value == "RULE"

    def test_audit_actor_and_entity_types(self):
        assert AuditActorType.SYSTEM.value == "SYSTEM"
        assert AuditActorType.API_CLIENT.value == "API_CLIENT"
        assert AuditEntityType.TRANSACTION.value == "TRANSACTION"
        assert AuditEntityType.RISK_EVALUATION.value == "RISK_EVALUATION"
