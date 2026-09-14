"""
Standard Risk Rule Catalog Module for the AI Fraud Detection & Risk Intelligence Platform.

Defines the verified, production-aligned canonical business rules catalog (STANDARD_RULES)
and factory function (get_standard_rule_catalog).

Design Principles:
- Uses strictly verified columns from PREDICTIVE_FEATURE_COLUMNS.
- Avoids unconfirmed deterministic hard blocks (all former hard-block rules revised to MONITOR based on empirical OOT analysis).
- Preserves high-precision REVIEW rules (velocity burst and spending deviation).
- Includes provisional empirical monitoring thresholds (e.g. 140.0 km for geographic distance).
- Deeply immutable, deterministic priority ordering (sorted by priority ascending, then rule_id).
"""

from typing import Tuple

from ml.risk_engine.config import (
    RuleOutcome,
    RuleType,
    RuleOperator,
)
from ml.risk_engine.rules import (
    RiskRule,
)

# Canonical tuple of revised standard business rules in deterministic priority order
STANDARD_RULES: Tuple[RiskRule, ...] = (
    RiskRule(
        rule_id="RULE_VELOCITY_BURST_REVIEW",
        description="Route transactions with unusually high one-hour transaction velocity to manual review.",
        feature_name="txn_count_1h",
        operator=RuleOperator.GREATER_THAN,
        comparison_value=4.0,
        outcome=RuleOutcome.REVIEW,
        rule_type=RuleType.VELOCITY,
        priority=30,
    ),
    RiskRule(
        rule_id="RULE_AMT_ZSCORE_DEVIATION_REVIEW",
        description="Route transactions with extreme deviation from historical spending behavior to manual review.",
        feature_name="amount_zscore",
        operator=RuleOperator.GREATER_THAN,
        comparison_value=5.0,
        outcome=RuleOutcome.REVIEW,
        rule_type=RuleType.AMOUNT,
        priority=40,
    ),
    RiskRule(
        rule_id="RULE_AMT_EXTREME_MONITOR",
        description="Monitor unusually large transaction amounts without automatically blocking the transaction.",
        feature_name="amount",
        operator=RuleOperator.GREATER_THAN,
        comparison_value=5000.0,
        outcome=RuleOutcome.MONITOR,
        rule_type=RuleType.AMOUNT,
        priority=50,
    ),
    RiskRule(
        rule_id="RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR",
        description="Record large transaction amounts for compliance and audit monitoring without changing the ML decision.",
        feature_name="amount",
        operator=RuleOperator.GREATER_THAN,
        comparison_value=3000.0,
        outcome=RuleOutcome.MONITOR,
        rule_type=RuleType.COMPLIANCE,
        priority=50,
    ),
    RiskRule(
        rule_id="RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR",
        description="Monitor impossible-travel signals without automatically blocking transactions because e-commerce and remote processing can create misleading geographic speeds.",
        feature_name="is_impossible_travel_speed",
        operator=RuleOperator.IS_TRUE,
        comparison_value=True,
        outcome=RuleOutcome.MONITOR,
        rule_type=RuleType.GEOGRAPHY,
        priority=60,
    ),
    RiskRule(
        rule_id="RULE_GEO_EXTREME_DISTANCE_MONITOR",
        description="Monitor transactions in the upper tail of cardholder-to-merchant geographic distance (provisional threshold: 140.0 km).",
        feature_name="cardholder_merchant_distance_km",
        operator=RuleOperator.GREATER_THAN,
        comparison_value=140.0,
        outcome=RuleOutcome.MONITOR,
        rule_type=RuleType.GEOGRAPHY,
        priority=70,
    ),
)


def get_standard_rule_catalog() -> Tuple[RiskRule, ...]:
    """
    Return the immutable canonical standard rule catalog in deterministic evaluation order.

    The catalog contains 6 validated RiskRule instances:
    1. RULE_VELOCITY_BURST_REVIEW (Priority 30, REVIEW)
    2. RULE_AMT_ZSCORE_DEVIATION_REVIEW (Priority 40, REVIEW)
    3. RULE_AMT_EXTREME_MONITOR (Priority 50, MONITOR)
    4. RULE_COMPLIANCE_LARGE_AMOUNT_MONITOR (Priority 50, MONITOR)
    5. RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR (Priority 60, MONITOR)
    6. RULE_GEO_EXTREME_DISTANCE_MONITOR (Priority 70, MONITOR - provisional threshold: 140.0 km)

    Returns:
        Tuple[RiskRule, ...]: Immutable tuple of standard rules sorted deterministically by (priority, rule_id).
    """
    return STANDARD_RULES
