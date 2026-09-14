"""
Automated Unit Tests for Phase 6 Increment 5A: Deterministic Business Rules & Predicate Matching Engine.

Validates:
1. Enums: RuleType, RuleOutcome, RuleOperator string values and invalid parsing.
2. RiskRule: Construction, attribute validation, operator-specific comparison checks, immutability, to_dict, from_dict.
3. RuleMatch: Immutability, type validation, serialization.
4. RuleEngine: Construction, duplicate rule_id rejection, deterministic priority ordering, empty rule sets, to_dict, from_dict.
5. Predicate Operators: Numeric (>, <), Membership (in), Boolean Equality (is_true) under various dtypes.
6. Missing Feature & Boundary Handling: Missing keys, None, NaN, inf, boundary equality, type mismatches.
7. Immutability Guarantees: Input features preservation, configuration immutability, returned matches immutability.
"""

import math
from typing import Any
import pytest
import numpy as np
import pandas as pd

from ml.risk_engine.config import (
    RuleType,
    RuleOutcome,
    RuleOperator,
)
from ml.risk_engine.rules import (
    RiskRule,
    RuleMatch,
    RuleEngine,
)


# =====================================================================
# 1. Enums Tests
# =====================================================================

class TestRuleEnums:
    """Test suite for RuleType, RuleOutcome, and RuleOperator enums."""

    def test_rule_type_values(self) -> None:
        """Verify standard rule category types."""
        assert RuleType.VELOCITY == "VELOCITY"
        assert RuleType.AMOUNT == "AMOUNT"
        assert RuleType.GEOGRAPHY == "GEOGRAPHY"
        assert RuleType.DEVICE == "DEVICE"
        assert RuleType.COMPLIANCE == "COMPLIANCE"
        assert RuleType.CUSTOM == "CUSTOM"

    def test_rule_outcome_values(self) -> None:
        """Verify rule outcome action types."""
        assert RuleOutcome.BLOCK == "BLOCK"
        assert RuleOutcome.REVIEW == "REVIEW"
        assert RuleOutcome.MONITOR == "MONITOR"

    def test_rule_operator_values(self) -> None:
        """Verify supported deterministic comparison operators."""
        assert RuleOperator.GREATER_THAN == ">"
        assert RuleOperator.LESS_THAN == "<"
        assert RuleOperator.IN == "in"
        assert RuleOperator.IS_TRUE == "is_true"


# =====================================================================
# 2. RiskRule Definition & Validation Tests
# =====================================================================

class TestRiskRuleDefinition:
    """Test suite for RiskRule creation, immutability, and validation."""

    def test_valid_numeric_greater_than_rule(self) -> None:
        """Verify valid numeric greater-than rule creation."""
        rule = RiskRule(
            rule_id="RULE_HIGH_AMOUNT",
            description="Transaction amount exceeds 5000",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=5000.0,
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.AMOUNT,
            priority=10,
        )
        assert rule.rule_id == "RULE_HIGH_AMOUNT"
        assert rule.description == "Transaction amount exceeds 5000"
        assert rule.feature_name == "amount"
        assert rule.operator == RuleOperator.GREATER_THAN
        assert rule.comparison_value == 5000.0
        assert rule.outcome == RuleOutcome.BLOCK
        assert rule.rule_type == RuleType.AMOUNT
        assert rule.priority == 10

    def test_string_enum_parsing(self) -> None:
        """Verify string inputs for enums are parsed properly."""
        rule = RiskRule(
            rule_id="RULE_VELOCITY",
            description="Velocity limit breach",
            feature_name="velocity_1h",
            operator=">",
            comparison_value=10,
            outcome="REVIEW",
            rule_type="VELOCITY",
            priority=20,
        )
        assert rule.operator == RuleOperator.GREATER_THAN
        assert rule.outcome == RuleOutcome.REVIEW
        assert rule.rule_type == RuleType.VELOCITY

    def test_membership_in_rule_frozenset_conversion(self) -> None:
        """Verify membership rule converts comparison list to immutable frozenset."""
        rule = RiskRule(
            rule_id="RULE_BLOCKED_MCC",
            description="High risk merchant category code",
            feature_name="merchant_category",
            operator=RuleOperator.IN,
            comparison_value=["gambling", "crypto_exchange", "wire_transfer"],
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.COMPLIANCE,
        )
        assert isinstance(rule.comparison_value, frozenset)
        assert rule.comparison_value == frozenset({"gambling", "crypto_exchange", "wire_transfer"})

    def test_is_true_rule(self) -> None:
        """Verify boolean equality rule setup."""
        rule = RiskRule(
            rule_id="RULE_NEW_DEVICE",
            description="Transaction from previously unseen device",
            feature_name="is_new_device",
            operator=RuleOperator.IS_TRUE,
            comparison_value=True,
            outcome=RuleOutcome.REVIEW,
            rule_type=RuleType.DEVICE,
        )
        assert rule.operator == RuleOperator.IS_TRUE
        assert rule.comparison_value is True

    def test_rule_immutability(self) -> None:
        """Verify RiskRule fields cannot be mutated."""
        rule = RiskRule(
            rule_id="RULE_IMMUTABLE",
            description="Test rule",
            feature_name="amount",
            operator=RuleOperator.GREATER_THAN,
            comparison_value=100.0,
            outcome=RuleOutcome.REVIEW,
        )
        with pytest.raises(Exception):
            rule.comparison_value = 200.0  # type: ignore

    @pytest.mark.parametrize("invalid_id", ["", "   ", None, 123])
    def test_invalid_rule_id_rejection(self, invalid_id: Any) -> None:
        """Verify empty or non-string rule_id is rejected."""
        with pytest.raises((ValueError, TypeError)):
            RiskRule(
                rule_id=invalid_id,
                description="Test rule",
                feature_name="amount",
                operator=RuleOperator.GREATER_THAN,
                comparison_value=100.0,
                outcome=RuleOutcome.REVIEW,
            )

    @pytest.mark.parametrize("invalid_desc", ["", "   ", None, 456])
    def test_invalid_description_rejection(self, invalid_desc: Any) -> None:
        """Verify empty or non-string description is rejected."""
        with pytest.raises((ValueError, TypeError)):
            RiskRule(
                rule_id="RULE_001",
                description=invalid_desc,
                feature_name="amount",
                operator=RuleOperator.GREATER_THAN,
                comparison_value=100.0,
                outcome=RuleOutcome.REVIEW,
            )

    @pytest.mark.parametrize("invalid_feature", ["", "   ", None, [1, 2]])
    def test_invalid_feature_name_rejection(self, invalid_feature: Any) -> None:
        """Verify empty or non-string feature_name is rejected."""
        with pytest.raises((ValueError, TypeError)):
            RiskRule(
                rule_id="RULE_001",
                description="Valid desc",
                feature_name=invalid_feature,
                operator=RuleOperator.GREATER_THAN,
                comparison_value=100.0,
                outcome=RuleOutcome.REVIEW,
            )

    @pytest.mark.parametrize("invalid_op", ["==", ">=", "<=", "regex", "contains", "!=", 123])
    def test_unsupported_operator_rejection(self, invalid_op: Any) -> None:
        """Verify unsupported operators raise ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            RiskRule(
                rule_id="RULE_001",
                description="Valid desc",
                feature_name="amount",
                operator=invalid_op,
                comparison_value=100.0,
                outcome=RuleOutcome.REVIEW,
            )

    @pytest.mark.parametrize("invalid_outcome", ["ALLOW", "DENY", "PASSTHROUGH", 999])
    def test_invalid_outcome_rejection(self, invalid_outcome: Any) -> None:
        """Verify invalid outcomes raise ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            RiskRule(
                rule_id="RULE_001",
                description="Valid desc",
                feature_name="amount",
                operator=RuleOperator.GREATER_THAN,
                comparison_value=100.0,
                outcome=invalid_outcome,
            )

    @pytest.mark.parametrize("invalid_priority", ["high", 1.5, True, False, None])
    def test_invalid_priority_rejection(self, invalid_priority: Any) -> None:
        """Verify non-integer priorities raise TypeError."""
        with pytest.raises(TypeError):
            RiskRule(
                rule_id="RULE_001",
                description="Valid desc",
                feature_name="amount",
                operator=RuleOperator.GREATER_THAN,
                comparison_value=100.0,
                outcome=RuleOutcome.REVIEW,
                priority=invalid_priority,
            )

    @pytest.mark.parametrize("invalid_numeric_val", [True, False, "500", [500], None, float("nan"), float("inf")])
    def test_invalid_numeric_comparison_value_rejection(self, invalid_numeric_val: Any) -> None:
        """Verify boolean, non-numeric, or non-finite values are rejected for > and <."""
        with pytest.raises((TypeError, ValueError)):
            RiskRule(
                rule_id="RULE_001",
                description="Valid desc",
                feature_name="amount",
                operator=RuleOperator.GREATER_THAN,
                comparison_value=invalid_numeric_val,
                outcome=RuleOutcome.REVIEW,
            )

    @pytest.mark.parametrize("invalid_container", ["single_str", 123, True, [], set(), None])
    def test_invalid_in_comparison_value_rejection(self, invalid_container: Any) -> None:
        """Verify empty container or non-container types are rejected for 'in' operator."""
        with pytest.raises((TypeError, ValueError)):
            RiskRule(
                rule_id="RULE_001",
                description="Valid desc",
                feature_name="merchant_category",
                operator=RuleOperator.IN,
                comparison_value=invalid_container,
                outcome=RuleOutcome.BLOCK,
            )

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """Verify serialization to/from dictionary."""
        original = RiskRule(
            rule_id="RULE_MCC",
            description="Blocked categories",
            feature_name="mcc",
            operator=RuleOperator.IN,
            comparison_value=["A", "B", "C"],
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.COMPLIANCE,
            priority=5,
        )
        d = original.to_dict()
        assert d == {
            "rule_id": "RULE_MCC",
            "description": "Blocked categories",
            "feature_name": "mcc",
            "operator": "in",
            "comparison_value": ["A", "B", "C"],
            "outcome": "BLOCK",
            "rule_type": "COMPLIANCE",
            "priority": 5,
        }
        reconstituted = RiskRule.from_dict(d)
        assert reconstituted == original


# =====================================================================
# 3. RuleEngine Construction & Ordering Tests
# =====================================================================

class TestRuleEngineConstruction:
    """Test suite for RuleEngine initialization, priority ordering, and validation."""

    def test_empty_rule_engine(self) -> None:
        """Verify empty rule engine initialization."""
        engine = RuleEngine()
        assert len(engine) == 0
        assert engine.rules == ()
        assert engine.evaluate({"amount": 1000.0}) == ()

    def test_deterministic_priority_sorting(self) -> None:
        """Verify rules are sorted deterministically by (priority, rule_id)."""
        r1 = RiskRule("R_LOW_PRIO", "desc", "f1", RuleOperator.GREATER_THAN, 10.0, RuleOutcome.MONITOR, priority=100)
        r2 = RiskRule("R_HIGH_PRIO_B", "desc", "f2", RuleOperator.GREATER_THAN, 20.0, RuleOutcome.BLOCK, priority=10)
        r3 = RiskRule("R_HIGH_PRIO_A", "desc", "f3", RuleOperator.GREATER_THAN, 30.0, RuleOutcome.REVIEW, priority=10)
        r4 = RiskRule("R_MED_PRIO", "desc", "f4", RuleOperator.GREATER_THAN, 40.0, RuleOutcome.REVIEW, priority=50)

        # Pass in unordered sequence
        engine = RuleEngine([r1, r2, r3, r4])
        assert [r.rule_id for r in engine.rules] == ["R_HIGH_PRIO_A", "R_HIGH_PRIO_B", "R_MED_PRIO", "R_LOW_PRIO"]

    def test_duplicate_rule_id_rejection(self) -> None:
        """Verify duplicate rule IDs raise ValueError."""
        r1 = RiskRule("R_DUPLICATE", "desc 1", "f1", RuleOperator.GREATER_THAN, 10.0, RuleOutcome.MONITOR)
        r2 = RiskRule("R_DUPLICATE", "desc 2", "f2", RuleOperator.GREATER_THAN, 20.0, RuleOutcome.BLOCK)
        with pytest.raises(ValueError, match="Duplicate rule_id detected"):
            RuleEngine([r1, r2])

    def test_invalid_rules_sequence_type(self) -> None:
        """Verify non-sequence rules argument raises TypeError."""
        with pytest.raises(TypeError, match="rules must be a sequence"):
            RuleEngine("not_a_sequence")  # type: ignore

    def test_invalid_element_in_rules_sequence(self) -> None:
        """Verify non-RiskRule elements in sequence raise TypeError."""
        r1 = RiskRule("R1", "desc", "f1", RuleOperator.GREATER_THAN, 10.0, RuleOutcome.MONITOR)
        with pytest.raises(TypeError, match="Every element in rules must be a RiskRule"):
            RuleEngine([r1, "invalid_rule"])  # type: ignore

    def test_engine_to_dict_and_from_dict(self) -> None:
        """Verify RuleEngine serialization roundtrip."""
        r1 = RiskRule("R1", "desc 1", "amount", RuleOperator.GREATER_THAN, 5000.0, RuleOutcome.BLOCK, priority=1)
        r2 = RiskRule("R2", "desc 2", "mcc", RuleOperator.IN, ["gambling"], RuleOutcome.REVIEW, priority=2)
        engine = RuleEngine([r2, r1])

        d = engine.to_dict()
        assert len(d["rules"]) == 2
        assert d["rules"][0]["rule_id"] == "R1"  # Sorted by priority
        assert d["rules"][1]["rule_id"] == "R2"

        reconstituted = RuleEngine.from_dict(d)
        assert len(reconstituted) == 2
        assert reconstituted.rules == engine.rules


# =====================================================================
# 4. Predicate Evaluation & Matching Tests
# =====================================================================

class TestPredicateEvaluation:
    """Test suite for evaluating operators against various inputs and edge cases."""

    @pytest.fixture
    def standard_engine(self) -> RuleEngine:
        """Fixture with a representative set of rules across all supported operators."""
        return RuleEngine([
            RiskRule(
                rule_id="RULE_AMOUNT_GT_1000",
                description="Amount exceeds 1000",
                feature_name="amount",
                operator=RuleOperator.GREATER_THAN,
                comparison_value=1000.0,
                outcome=RuleOutcome.BLOCK,
                priority=10,
            ),
            RiskRule(
                rule_id="RULE_BALANCE_LT_50",
                description="Account balance below 50",
                feature_name="balance",
                operator=RuleOperator.LESS_THAN,
                comparison_value=50.0,
                outcome=RuleOutcome.REVIEW,
                priority=20,
            ),
            RiskRule(
                rule_id="RULE_HIGH_RISK_MCC",
                description="High risk merchant category",
                feature_name="merchant_category",
                operator=RuleOperator.IN,
                comparison_value=["crypto", "wire_transfer"],
                outcome=RuleOutcome.BLOCK,
                priority=30,
            ),
            RiskRule(
                rule_id="RULE_FOREIGN_TRANSACTION",
                description="Foreign IP flag is set",
                feature_name="is_foreign_ip",
                operator=RuleOperator.IS_TRUE,
                comparison_value=True,
                outcome=RuleOutcome.MONITOR,
                priority=40,
            ),
        ])

    def test_numeric_greater_than_boundary(self, standard_engine: RuleEngine) -> None:
        """Verify strict inequality behavior for greater than (>)."""
        # Strictly below: False
        assert len(standard_engine.evaluate({"amount": 999.99})) == 0

        # Exact boundary: False
        assert len(standard_engine.evaluate({"amount": 1000.0})) == 0

        # Strictly above: True
        matches = standard_engine.evaluate({"amount": 1000.01})
        assert len(matches) == 1
        assert matches[0].rule_id == "RULE_AMOUNT_GT_1000"
        assert matches[0].outcome == RuleOutcome.BLOCK
        assert matches[0].feature_value == 1000.01

    def test_numeric_less_than_boundary(self, standard_engine: RuleEngine) -> None:
        """Verify strict inequality behavior for less than (<)."""
        # Strictly above: False
        assert len(standard_engine.evaluate({"balance": 50.01})) == 0

        # Exact boundary: False
        assert len(standard_engine.evaluate({"balance": 50.0})) == 0

        # Strictly below: True
        matches = standard_engine.evaluate({"balance": 49.99})
        assert len(matches) == 1
        assert matches[0].rule_id == "RULE_BALANCE_LT_50"
        assert matches[0].outcome == RuleOutcome.REVIEW

    def test_membership_in_matching(self, standard_engine: RuleEngine) -> None:
        """Verify membership in set."""
        # Member: True
        matches_crypto = standard_engine.evaluate({"merchant_category": "crypto"})
        assert len(matches_crypto) == 1
        assert matches_crypto[0].rule_id == "RULE_HIGH_RISK_MCC"

        matches_wire = standard_engine.evaluate({"merchant_category": "wire_transfer"})
        assert len(matches_wire) == 1

        # Non-member: False
        assert len(standard_engine.evaluate({"merchant_category": "grocery"})) == 0

    @pytest.mark.parametrize("true_val", [True, 1, "true", "TRUE", "True", "1"])
    def test_is_true_truthy_values(self, standard_engine: RuleEngine, true_val: Any) -> None:
        """Verify is_true matches all recognized truthy representations."""
        matches = standard_engine.evaluate({"is_foreign_ip": true_val})
        assert len(matches) == 1
        assert matches[0].rule_id == "RULE_FOREIGN_TRANSACTION"

    @pytest.mark.parametrize("false_val", [False, 0, "false", "FALSE", "0", None, 2, "random_str"])
    def test_is_true_falsy_values(self, standard_engine: RuleEngine, false_val: Any) -> None:
        """Verify is_true does not match falsy or non-true values."""
        assert len(standard_engine.evaluate({"is_foreign_ip": false_val})) == 0

    def test_multiple_rules_matched_in_deterministic_order(self, standard_engine: RuleEngine) -> None:
        """Verify multiple triggered rules return in priority order."""
        payload = {
            "amount": 2500.0,            # Triggers RULE_AMOUNT_GT_1000 (priority 10)
            "balance": 10.0,             # Triggers RULE_BALANCE_LT_50 (priority 20)
            "merchant_category": "crypto", # Triggers RULE_HIGH_RISK_MCC (priority 30)
            "is_foreign_ip": True,       # Triggers RULE_FOREIGN_TRANSACTION (priority 40)
        }
        matches = standard_engine.evaluate(payload)
        assert len(matches) == 4
        assert [m.rule_id for m in matches] == [
            "RULE_AMOUNT_GT_1000",
            "RULE_BALANCE_LT_50",
            "RULE_HIGH_RISK_MCC",
            "RULE_FOREIGN_TRANSACTION",
        ]


# =====================================================================
# 5. Missing Features & Robustness Tests
# =====================================================================

class TestMissingFeaturesAndRobustness:
    """Test suite for missing features, None values, NaN, inf, and input types."""

    def test_missing_feature_evaluates_to_false(self) -> None:
        """Verify missing feature keys do not trigger rules and do not raise errors."""
        engine = RuleEngine([
            RiskRule("R1", "desc", "amount", RuleOperator.GREATER_THAN, 100.0, RuleOutcome.BLOCK),
            RiskRule("R2", "desc", "mcc", RuleOperator.IN, ["gambling"], RuleOutcome.BLOCK),
        ])
        # Empty payload
        assert engine.evaluate({}) == ()
        # Unrelated feature payload
        assert engine.evaluate({"unrelated_col": 999.0}) == ()

    def test_none_value_evaluates_to_false(self) -> None:
        """Verify None values for feature do not trigger rules."""
        engine = RuleEngine([
            RiskRule("R1", "desc", "amount", RuleOperator.GREATER_THAN, 100.0, RuleOutcome.BLOCK),
            RiskRule("R2", "desc", "amount", RuleOperator.LESS_THAN, 100.0, RuleOutcome.BLOCK),
            RiskRule("R3", "desc", "mcc", RuleOperator.IN, ["gambling"], RuleOutcome.BLOCK),
            RiskRule("R4", "desc", "flag", RuleOperator.IS_TRUE, True, RuleOutcome.BLOCK),
        ])
        payload = {"amount": None, "mcc": None, "flag": None}
        assert engine.evaluate(payload) == ()

    def test_nan_and_inf_numeric_features_evaluate_to_false(self) -> None:
        """Verify NaN and +/- inf features do not trigger numeric rules."""
        engine = RuleEngine([
            RiskRule("R1", "desc", "amount", RuleOperator.GREATER_THAN, 100.0, RuleOutcome.BLOCK),
            RiskRule("R2", "desc", "amount", RuleOperator.LESS_THAN, 100.0, RuleOutcome.BLOCK),
        ])
        assert engine.evaluate({"amount": float("nan")}) == ()
        assert engine.evaluate({"amount": float("inf")}) == ()
        assert engine.evaluate({"amount": float("-inf")}) == ()

    def test_type_mismatch_numeric_operator_safe_handling(self) -> None:
        """Verify passing non-numeric strings or bools to numeric rules does not trigger or crash."""
        engine = RuleEngine([
            RiskRule("R1", "desc", "amount", RuleOperator.GREATER_THAN, 100.0, RuleOutcome.BLOCK),
            RiskRule("R2", "desc", "amount", RuleOperator.LESS_THAN, 100.0, RuleOutcome.BLOCK),
        ])
        # String passed to numeric rule
        assert engine.evaluate({"amount": "not_a_number"}) == ()
        # Boolean passed to numeric rule (must not evaluate True as 1 or False as 0)
        assert engine.evaluate({"amount": True}) == ()
        assert engine.evaluate({"amount": False}) == ()

    def test_pandas_series_input_compatibility(self) -> None:
        """Verify RuleEngine accepts a pd.Series input and produces identical results."""
        engine = RuleEngine([
            RiskRule("R1", "desc", "amount", RuleOperator.GREATER_THAN, 500.0, RuleOutcome.BLOCK),
            RiskRule("R2", "desc", "is_new", RuleOperator.IS_TRUE, True, RuleOutcome.REVIEW),
        ])
        series = pd.Series({"amount": 1000.0, "is_new": 1, "other_col": "val"})
        matches = engine.evaluate(series)
        assert len(matches) == 2
        assert matches[0].rule_id == "R1"
        assert matches[1].rule_id == "R2"

    def test_invalid_features_payload_type(self) -> None:
        """Verify non-mapping/non-Series payload raises TypeError."""
        engine = RuleEngine()
        with pytest.raises(TypeError, match="features must be a Mapping"):
            engine.evaluate([1, 2, 3])  # type: ignore

    def test_input_features_immutability(self) -> None:
        """Verify evaluating does not mutate the caller's input dictionary."""
        engine = RuleEngine([
            RiskRule("R1", "desc", "amount", RuleOperator.GREATER_THAN, 100.0, RuleOutcome.BLOCK)
        ])
        payload = {"amount": 500.0, "user_id": "usr_123"}
        payload_copy = dict(payload)
        _ = engine.evaluate(payload)
        assert payload == payload_copy

    def test_rule_match_immutability_and_serialization(self) -> None:
        """Verify RuleMatch is frozen dataclass and produces clean to_dict."""
        match = RuleMatch(
            rule_id="R1",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.AMOUNT,
            feature_name="amount",
            feature_value=1500.50,
            operator=RuleOperator.GREATER_THAN,
            comparison_value=1000.0,
            description="Amount above 1000",
            priority=10,
        )
        with pytest.raises(Exception):
            match.outcome = RuleOutcome.REVIEW  # type: ignore

        d = match.to_dict()
        assert d == {
            "rule_id": "R1",
            "outcome": "BLOCK",
            "rule_type": "AMOUNT",
            "feature_name": "amount",
            "feature_value": 1500.5,
            "operator": ">",
            "comparison_value": 1000.0,
            "description": "Amount above 1000",
            "priority": 10,
        }
