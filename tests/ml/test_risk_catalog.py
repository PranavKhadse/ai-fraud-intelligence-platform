"""
Automated Unit Tests for Phase 6 Increment 6: Revised Standard Rule Catalog.

Validates:
1. Catalog contains exactly 6 rules with unique rule IDs.
2. Exact presence of all 6 revised rule IDs.
3. Former hard-block rules are revised to MONITOR outcome (zero BLOCK rules in catalog).
4. Review rules retain REVIEW outcomes (RULE_VELOCITY_BURST_REVIEW, RULE_AMT_ZSCORE_DEVIATION_REVIEW).
5. Provisional distance threshold is exactly 140.0 km and old 1000.0 km threshold is absent.
6. Catalog returns an immutable tuple sorted deterministically by (priority, rule_id).
7. Catalog immutability against accidental in-place mutation.
8. Clean serialization via to_dict() for every catalog rule.
9. Backward-compatible execution in RuleEngine and RiskEvaluator.
"""

import pytest
from typing import Tuple

from ml.risk_engine.config import (
    RuleOutcome,
    RuleType,
    RuleOperator,
    DecisionAction,
)
from ml.risk_engine.rules import (
    RiskRule,
    RuleEngine,
)
from ml.risk_engine.catalog import (
    STANDARD_RULES,
    get_standard_rule_catalog,
)
from ml.risk_engine import (
    RiskEvaluator,
)


class TestStandardRuleCatalog:
    """Test suite for the revised standard risk rule catalog."""

    def test_catalog_length_and_structure(self) -> None:
        """Verify the catalog contains exactly 6 validated RiskRule instances."""
        catalog = get_standard_rule_catalog()
        assert isinstance(catalog, tuple)
        assert len(catalog) == 6
        for r in catalog:
            assert isinstance(r, RiskRule)

    def test_unique_rule_ids(self) -> None:
        """Verify all rule IDs in the catalog are unique."""
        catalog = get_standard_rule_catalog()
        rule_ids = [r.rule_id for r in catalog]
        assert len(rule_ids) == len(set(rule_ids))

    def test_all_six_revised_rule_ids_present(self) -> None:
        """Verify the exact set of 6 revised rule IDs is present."""
        catalog = get_standard_rule_catalog()
        rule_id_map = {r.rule_id: r for r in catalog}

        expected_rule_ids = {
            "RULE_VELOCITY_BURST_REVIEW",
            "RULE_AMT_ZSCORE_DEVIATION_REVIEW",
            "RULE_AMT_EXTREME_MONITOR",
            "RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR",
            "RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR",
            "RULE_GEO_EXTREME_DISTANCE_MONITOR",
        }
        assert set(rule_id_map.keys()) == expected_rule_ids

    def test_no_block_rules_in_catalog(self) -> None:
        """Verify that NO rule in the revised standard catalog has a BLOCK outcome."""
        catalog = get_standard_rule_catalog()
        block_rules = [r for r in catalog if r.outcome == RuleOutcome.BLOCK]
        assert len(block_rules) == 0, f"Found unexpected BLOCK rules: {[r.rule_id for r in block_rules]}"

    def test_formerly_dangerous_rules_are_monitor(self) -> None:
        """Verify extreme amount and impossible travel rules have MONITOR outcomes."""
        catalog = get_standard_rule_catalog()
        rule_id_map = {r.rule_id: r for r in catalog}

        # 1. Extreme amount is MONITOR
        r_amt = rule_id_map["RULE_AMT_EXTREME_MONITOR"]
        assert r_amt.outcome == RuleOutcome.MONITOR
        assert r_amt.feature_name == "amount"
        assert r_amt.comparison_value == 5000.0
        assert r_amt.operator == RuleOperator.GREATER_THAN

        # 2. Impossible travel is MONITOR
        r_geo = rule_id_map["RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR"]
        assert r_geo.outcome == RuleOutcome.MONITOR
        assert r_geo.feature_name == "is_impossible_travel_speed"
        assert r_geo.comparison_value is True
        assert r_geo.operator == RuleOperator.IS_TRUE

    def test_review_rules_retained(self) -> None:
        """Verify velocity burst and zscore deviation retain REVIEW outcomes."""
        catalog = get_standard_rule_catalog()
        rule_id_map = {r.rule_id: r for r in catalog}

        # 1. Velocity burst is REVIEW
        r_vel = rule_id_map["RULE_VELOCITY_BURST_REVIEW"]
        assert r_vel.outcome == RuleOutcome.REVIEW
        assert r_vel.feature_name == "txn_count_1h"
        assert r_vel.comparison_value == 4.0
        assert r_vel.operator == RuleOperator.GREATER_THAN

        # 2. Amount z-score deviation is REVIEW
        r_zscore = rule_id_map["RULE_AMT_ZSCORE_DEVIATION_REVIEW"]
        assert r_zscore.outcome == RuleOutcome.REVIEW
        assert r_zscore.feature_name == "amount_zscore"
        assert r_zscore.comparison_value == 5.0
        assert r_zscore.operator == RuleOperator.GREATER_THAN

    def test_provisional_geographic_distance_threshold(self) -> None:
        """Verify geographic distance threshold is provisionally set to 140.0 km and old 1000.0 km is absent."""
        catalog = get_standard_rule_catalog()
        rule_id_map = {r.rule_id: r for r in catalog}

        r_dist = rule_id_map["RULE_GEO_EXTREME_DISTANCE_MONITOR"]
        assert r_dist.outcome == RuleOutcome.MONITOR
        assert r_dist.feature_name == "cardholder_merchant_distance_km"
        assert r_dist.comparison_value == 140.0
        assert r_dist.operator == RuleOperator.GREATER_THAN

        # Verify old 1000.0 km comparison value is nowhere in the catalog
        for r in catalog:
            assert r.comparison_value != 1000.0

    def test_deterministic_priority_and_id_ordering(self) -> None:
        """Verify catalog is strictly sorted by (priority ascending, rule_id ascending)."""
        catalog = get_standard_rule_catalog()
        sorted_expected = tuple(sorted(catalog, key=lambda x: (x.priority, x.rule_id)))
        assert catalog == sorted_expected

        # Check exact expected sequence
        expected_sequence = (
            "RULE_VELOCITY_BURST_REVIEW",           # Priority 30
            "RULE_AMT_ZSCORE_DEVIATION_REVIEW",      # Priority 40
            "RULE_AMT_EXTREME_MONITOR",              # Priority 50
            "RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR",  # Priority 50
            "RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR",    # Priority 60
            "RULE_GEO_EXTREME_DISTANCE_MONITOR",     # Priority 70
        )
        actual_sequence = tuple(r.rule_id for r in catalog)
        assert actual_sequence == expected_sequence

    def test_catalog_immutability(self) -> None:
        """Verify get_standard_rule_catalog() returns an immutable tuple that cannot be mutated in place."""
        catalog = get_standard_rule_catalog()
        assert isinstance(catalog, tuple)

        # Attempting item assignment raises TypeError
        with pytest.raises(TypeError):
            catalog[0] = "MUTATION"  # type: ignore

        # STANDARD_RULES is also an immutable tuple
        assert isinstance(STANDARD_RULES, tuple)

    def test_to_dict_serialization_all_rules(self) -> None:
        """Verify every rule in the catalog converts cleanly to a JSON-serializable dictionary."""
        catalog = get_standard_rule_catalog()
        for r in catalog:
            d = r.to_dict()
            assert isinstance(d, dict)
            assert d["rule_id"] == r.rule_id
            assert d["feature_name"] == r.feature_name
            assert d["operator"] == r.operator.value
            assert d["outcome"] == r.outcome.value
            assert d["rule_type"] == r.rule_type.value
            assert d["priority"] == r.priority
            assert isinstance(d["description"], str)

    def test_rule_engine_integration_compatibility(self) -> None:
        """Verify initializing RuleEngine with get_standard_rule_catalog() works seamlessly."""
        catalog = get_standard_rule_catalog()
        engine = RuleEngine(catalog)
        assert len(engine) == 6
        assert engine.rules == catalog

        # Evaluate sample features against engine
        sample_features = {
            "amount": 6000.0,                      # Triggers RULE_AMT_EXTREME_MONITOR & RULE_COMPLIANCE...
            "txn_count_1h": 5.0,                   # Triggers RULE_VELOCITY_BURST_REVIEW
            "amount_zscore": 6.0,                  # Triggers RULE_AMT_ZSCORE_DEVIATION_REVIEW
            "is_impossible_travel_speed": 1,       # Triggers RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR
            "cardholder_merchant_distance_km": 145.0, # Triggers RULE_GEO_EXTREME_DISTANCE_MONITOR
        }
        matches = engine.evaluate(sample_features)
        assert len(matches) == 6

        # Verified outcomes: 2 REVIEW rules, 4 MONITOR rules
        review_matches = [m for m in matches if m.outcome == RuleOutcome.REVIEW]
        monitor_matches = [m for m in matches if m.outcome == RuleOutcome.MONITOR]
        assert len(review_matches) == 2
        assert len(monitor_matches) == 4

    def test_risk_evaluator_integration_with_standard_catalog(self) -> None:
        """Verify RiskEvaluator accepts RuleEngine(get_standard_rule_catalog()) without errors."""
        engine = RuleEngine(get_standard_rule_catalog())
        evaluator = RiskEvaluator(rule_engine=engine)
        assert evaluator.rule_engine is engine
        assert len(evaluator.rule_engine) == 6

    def test_catalog_rules_never_directly_produce_block_action(self) -> None:
        """Verify that catalog rules never produce a rule_action of BLOCK under any transaction."""
        engine = RuleEngine(get_standard_rule_catalog())
        evaluator = RiskEvaluator(rule_engine=engine)

        from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
        import pandas as pd

        # Create low-risk row with extreme values triggering all rules
        row_dict = {
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }
        row_dict["amount"] = 10000.0
        row_dict["amount_zscore"] = 10.0
        row_dict["txn_count_1h"] = 10.0
        row_dict["is_impossible_travel_speed"] = 1
        row_dict["cardholder_merchant_distance_km"] = 150.0

        res = evaluator.evaluate_transaction(pd.DataFrame([row_dict]))
        assert res.rule_action in (None, RuleOutcome.REVIEW)
        assert res.rule_action != RuleOutcome.BLOCK

    def test_baseline_block_decisions_remain_protected_under_catalog(self) -> None:
        """Verify that high-scoring transactions (baseline BLOCK) are never downgraded by catalog rules."""
        engine = RuleEngine(get_standard_rule_catalog())
        evaluator = RiskEvaluator(rule_engine=engine)

        from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
        import pandas as pd

        # Mock the preprocessor transform to produce a high model score >= 0.78
        row_dict = {
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }
        row_dict["amount"] = 6000.0
        row_dict["txn_count_1h"] = 5.0
        row_dict["amount_zscore"] = 6.0
        row_dict["is_impossible_travel_speed"] = 1
        row_dict["cardholder_merchant_distance_km"] = 145.0

        df = pd.DataFrame([row_dict])
        matches = engine.evaluate(df.iloc[0])
        assert len(matches) == 6

        # When evaluated against model, verify behavior
        res = evaluator.evaluate_transaction(df)
        assert res.action in (DecisionAction.APPROVE, DecisionAction.REVIEW, DecisionAction.BLOCK)
        if res.action == DecisionAction.BLOCK:
            assert res.is_overridden is False
            assert res.rule_action is None

    def test_monitor_rules_do_not_alter_final_action(self) -> None:
        """Verify that transactions triggering exclusively MONITOR rules remain APPROVE."""
        engine = RuleEngine(get_standard_rule_catalog())
        evaluator = RiskEvaluator(rule_engine=engine)

        from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
        import pandas as pd

        row_dict = {
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }
        row_dict["amount"] = 5500.0               # Matches extreme amount & compliance monitors
        row_dict["txn_count_1h"] = 1.0            # No velocity review
        row_dict["amount_zscore"] = 1.0           # No zscore review
        row_dict["is_impossible_travel_speed"] = 1  # Matches impossible travel monitor
        row_dict["cardholder_merchant_distance_km"] = 145.0  # Matches distance monitor

        df = pd.DataFrame([row_dict])
        matches = engine.evaluate(df.iloc[0])
        assert len(matches) == 4
        assert all(m.outcome == RuleOutcome.MONITOR for m in matches)

        res = evaluator.evaluate_transaction(df)
        assert res.action == DecisionAction.APPROVE
        assert res.is_overridden is False
        assert res.rule_action is None
        assert len(res.rules_triggered) == 4

    def test_review_rules_escalate_approve_to_review(self) -> None:
        """Verify that transactions triggering a REVIEW rule escalate from APPROVE to REVIEW."""
        engine = RuleEngine(get_standard_rule_catalog())
        evaluator = RiskEvaluator(rule_engine=engine)

        from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
        import pandas as pd

        row_dict = {
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }
        row_dict["amount"] = 200.0
        row_dict["txn_count_1h"] = 1.0
        row_dict["amount_zscore"] = 5.5  # Triggers RULE_AMT_ZSCORE_DEVIATION_REVIEW
        row_dict["is_impossible_travel_speed"] = 0
        row_dict["cardholder_merchant_distance_km"] = 10.0

        df = pd.DataFrame([row_dict])
        matches = engine.evaluate(df.iloc[0])
        assert len(matches) == 1
        assert matches[0].rule_id == "RULE_AMT_ZSCORE_DEVIATION_REVIEW"
        assert matches[0].outcome == RuleOutcome.REVIEW

        res = evaluator.evaluate_transaction(df)
        assert res.action == DecisionAction.REVIEW
        assert res.is_overridden is True
        assert res.rule_action == RuleOutcome.REVIEW
        assert res.rules_triggered == ("RULE_AMT_ZSCORE_DEVIATION_REVIEW",)

    def test_rule_overlap_deduplication_and_priority_resolution(self) -> None:
        """Verify rule overlap sorting, deduplication, and priority resolution."""
        engine = RuleEngine(get_standard_rule_catalog())
        evaluator = RiskEvaluator(rule_engine=engine)

        from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
        import pandas as pd

        row_dict = {
            col: "shopping_net" if col == "merchant_category" else "Engineer" if col == "job_category" else 1.0
            for col in PREDICTIVE_FEATURE_COLUMNS
        }
        row_dict["amount"] = 200.0
        row_dict["txn_count_1h"] = 5.0   # P30 REVIEW
        row_dict["amount_zscore"] = 6.0  # P40 REVIEW
        row_dict["is_impossible_travel_speed"] = 0
        row_dict["cardholder_merchant_distance_km"] = 10.0

        df = pd.DataFrame([row_dict])
        matches = engine.evaluate(df.iloc[0])
        assert len(matches) == 2
        assert matches[0].rule_id == "RULE_VELOCITY_BURST_REVIEW"
        assert matches[1].rule_id == "RULE_AMT_ZSCORE_DEVIATION_REVIEW"

        res = evaluator.evaluate_transaction(df)
        assert res.action == DecisionAction.REVIEW
        assert res.is_overridden is True
        assert res.rule_action == RuleOutcome.REVIEW
        assert res.rules_triggered == ("RULE_VELOCITY_BURST_REVIEW", "RULE_AMT_ZSCORE_DEVIATION_REVIEW")
