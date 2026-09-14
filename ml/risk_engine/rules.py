"""
Deterministic Business Rules & Predicate Matching Engine.

Provides typed, immutable rule specifications (RiskRule) and a deterministic
matching engine (RuleEngine) for evaluating transaction features against business policies
without model scoring or mutation.
"""

from dataclasses import dataclass
import math
from typing import Dict, Any, List, Optional, Union, Mapping, Tuple, Sequence, Set, FrozenSet
import numpy as np
import pandas as pd

from ml.risk_engine.config import (
    RuleType,
    RuleOutcome,
    RuleOperator,
)


@dataclass(frozen=True)
class RiskRule:
    """
    Immutable specification of a deterministic business/heuristic rule.

    Attributes:
        rule_id: Unique, non-empty stable string identifier for the rule.
        description: Human-readable explanation of why the rule triggers.
        feature_name: Target feature name in transaction feature vector.
        operator: Predicate comparison operator (>, <, in, is_true).
        comparison_value: Immutable threshold or target set for comparison.
        outcome: Rule action/outcome (BLOCK, REVIEW, MONITOR).
        rule_type: Semantic categorization of rule (VELOCITY, AMOUNT, GEOGRAPHY, DEVICE, COMPLIANCE, CUSTOM).
        priority: Deterministic evaluation priority (lower value = higher precedence, default 100).
    """
    rule_id: str
    description: str
    feature_name: str
    operator: RuleOperator
    comparison_value: Any
    outcome: RuleOutcome
    rule_type: RuleType = RuleType.CUSTOM
    priority: int = 100

    def __post_init__(self) -> None:
        """Defensive validation of rule attributes and enforcement of deep immutability."""
        # 1. Validate rule_id
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError(f"rule_id must be a non-empty string, got {self.rule_id!r}")
        object.__setattr__(self, "rule_id", self.rule_id.strip())

        # 2. Validate description
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(f"description must be a non-empty string, got {self.description!r}")
        object.__setattr__(self, "description", self.description.strip())

        # 3. Validate feature_name
        if not isinstance(self.feature_name, str) or not self.feature_name.strip():
            raise ValueError(f"feature_name must be a non-empty string, got {self.feature_name!r}")
        object.__setattr__(self, "feature_name", self.feature_name.strip())

        # 4. Validate operator
        op = self.operator
        if isinstance(op, str):
            try:
                op = RuleOperator(op)
            except ValueError:
                valid_ops = [o.value for o in RuleOperator]
                raise ValueError(f"Unsupported operator '{op}'. Allowed operators: {valid_ops}")
        elif not isinstance(op, RuleOperator):
            raise TypeError(f"operator must be a RuleOperator enum or str, got {type(op).__name__}")
        object.__setattr__(self, "operator", op)

        # 5. Validate outcome
        out = self.outcome
        if isinstance(out, str):
            try:
                out = RuleOutcome(out)
            except ValueError:
                valid_outcomes = [o.value for o in RuleOutcome]
                raise ValueError(f"Invalid outcome '{out}'. Allowed outcomes: {valid_outcomes}")
        elif not isinstance(out, RuleOutcome):
            raise TypeError(f"outcome must be a RuleOutcome enum or str, got {type(out).__name__}")
        object.__setattr__(self, "outcome", out)

        # 6. Validate rule_type
        rt = self.rule_type
        if isinstance(rt, str):
            try:
                rt = RuleType(rt)
            except ValueError:
                valid_types = [t.value for t in RuleType]
                raise ValueError(f"Invalid rule_type '{rt}'. Allowed rule_types: {valid_types}")
        elif not isinstance(rt, RuleType):
            raise TypeError(f"rule_type must be a RuleType enum or str, got {type(rt).__name__}")
        object.__setattr__(self, "rule_type", rt)

        # 7. Validate priority
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError(f"priority must be an integer, got {type(self.priority).__name__}")

        # 8. Operator-specific comparison_value validation and immutability
        if op in (RuleOperator.GREATER_THAN, RuleOperator.LESS_THAN):
            if isinstance(self.comparison_value, bool) or not isinstance(self.comparison_value, (int, float)):
                raise TypeError(
                    f"comparison_value for operator '{op.value}' must be a numeric float or int, got {type(self.comparison_value).__name__}"
                )
            num_val = float(self.comparison_value)
            if not math.isfinite(num_val):
                raise ValueError(f"comparison_value for operator '{op.value}' must be finite, got {self.comparison_value}")
            object.__setattr__(self, "comparison_value", num_val)
        elif op == RuleOperator.IN:
            if isinstance(self.comparison_value, (str, bytes, Mapping)) or not isinstance(
                self.comparison_value, (set, frozenset, list, tuple)
            ):
                raise TypeError(
                    f"comparison_value for operator 'in' must be a container sequence (set, list, tuple, frozenset), got {type(self.comparison_value).__name__}"
                )
            if len(self.comparison_value) == 0:
                raise ValueError("comparison_value for operator 'in' cannot be empty.")
            # Convert to deep immutable frozenset
            object.__setattr__(self, "comparison_value", frozenset(self.comparison_value))
        elif op == RuleOperator.IS_TRUE:
            if self.comparison_value not in (None, True):
                if isinstance(self.comparison_value, bool) and not self.comparison_value:
                    raise ValueError("comparison_value for 'is_true' operator must be True or None.")
            object.__setattr__(self, "comparison_value", True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule specification to a clean JSON-serializable dictionary."""
        val = self.comparison_value
        if isinstance(val, (frozenset, set)):
            val = sorted(list(val), key=lambda x: str(x))
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "feature_name": self.feature_name,
            "operator": self.operator.value,
            "comparison_value": val,
            "outcome": self.outcome.value,
            "rule_type": self.rule_type.value,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskRule":
        """
        Reconstitute a validated RiskRule from a dictionary.

        Args:
            data: Dictionary containing rule parameters.

        Returns:
            RiskRule: Validated immutable rule instance.
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")
        return cls(
            rule_id=data["rule_id"],
            description=data["description"],
            feature_name=data["feature_name"],
            operator=data["operator"],
            comparison_value=data["comparison_value"],
            outcome=data["outcome"],
            rule_type=data.get("rule_type", RuleType.CUSTOM),
            priority=data.get("priority", 100),
        )


@dataclass(frozen=True)
class RuleMatch:
    """
    Immutable representation of a triggered rule evaluation against a transaction feature.

    Attributes:
        rule_id: Identifier of the rule that triggered.
        outcome: Action outcome of the rule (BLOCK, REVIEW, MONITOR).
        rule_type: Categorization category of the rule.
        feature_name: Name of the feature evaluated.
        feature_value: Actual value of the feature at evaluation time.
        operator: Predicate comparison operator.
        comparison_value: Configured comparison target.
        description: Human-readable explanation of the rule trigger.
        priority: Precedence priority of the matched rule.
    """
    rule_id: str
    outcome: RuleOutcome
    rule_type: RuleType
    feature_name: str
    feature_value: Any
    operator: RuleOperator
    comparison_value: Any
    description: str
    priority: int

    def __post_init__(self) -> None:
        """Defensive validation of rule match fields."""
        if not isinstance(self.rule_id, str):
            raise TypeError(f"rule_id must be a str, got {type(self.rule_id).__name__}")
        if not isinstance(self.outcome, RuleOutcome):
            raise TypeError(f"outcome must be a RuleOutcome enum, got {type(self.outcome).__name__}")
        if not isinstance(self.rule_type, RuleType):
            raise TypeError(f"rule_type must be a RuleType enum, got {type(self.rule_type).__name__}")
        if not isinstance(self.feature_name, str):
            raise TypeError(f"feature_name must be a str, got {type(self.feature_name).__name__}")
        if not isinstance(self.operator, RuleOperator):
            raise TypeError(f"operator must be a RuleOperator enum, got {type(self.operator).__name__}")
        if not isinstance(self.description, str):
            raise TypeError(f"description must be a str, got {type(self.description).__name__}")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError(f"priority must be an integer, got {type(self.priority).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule match outcome to a clean JSON-serializable dictionary."""
        val = self.comparison_value
        if isinstance(val, (frozenset, set)):
            val = sorted(list(val), key=lambda x: str(x))
        feat_val = self.feature_value
        if isinstance(feat_val, (np.floating, float)):
            feat_val = round(float(feat_val), 6)
        elif isinstance(feat_val, (np.integer, int)) and not isinstance(feat_val, bool):
            feat_val = int(feat_val)

        return {
            "rule_id": self.rule_id,
            "outcome": self.outcome.value,
            "rule_type": self.rule_type.value,
            "feature_name": self.feature_name,
            "feature_value": feat_val,
            "operator": self.operator.value,
            "comparison_value": val,
            "description": self.description,
            "priority": self.priority,
        }


class RuleEngine:
    """
    Deterministic Rule Engine that evaluates transaction feature mappings against
    configured RiskRules with strict immutability, priority ordering, and missing-feature safety.
    """

    def __init__(self, rules: Optional[Sequence[RiskRule]] = None) -> None:
        """
        Initialize the rule engine with a sequence of validated RiskRules.

        Args:
            rules: Optional sequence of RiskRule instances. If None, engine is empty.

        Raises:
            TypeError: If rules is not a sequence or contains non-RiskRule elements.
            ValueError: If duplicate rule_ids are detected.
        """
        if rules is None:
            self._rules: Tuple[RiskRule, ...] = ()
        else:
            if not isinstance(rules, (list, tuple)):
                raise TypeError(f"rules must be a sequence (list or tuple), got {type(rules).__name__}")
            rule_list: List[RiskRule] = []
            seen_ids: Set[str] = set()
            for r in rules:
                if not isinstance(r, RiskRule):
                    raise TypeError(f"Every element in rules must be a RiskRule instance, got {type(r).__name__}")
                if r.rule_id in seen_ids:
                    raise ValueError(f"Duplicate rule_id detected: '{r.rule_id}'. Rule IDs must be strictly unique.")
                seen_ids.add(r.rule_id)
                rule_list.append(r)
            # Deterministic ordering: sorted by priority ascending (lower int = higher precedence), then rule_id lexicographically
            self._rules = tuple(sorted(rule_list, key=lambda x: (x.priority, x.rule_id)))

    @property
    def rules(self) -> Tuple[RiskRule, ...]:
        """Return immutable tuple of configured rules in deterministic evaluation order."""
        return self._rules

    def __len__(self) -> int:
        """Return total count of active rules in the engine."""
        return len(self._rules)

    def evaluate(
        self, features: Union[Mapping[str, Any], pd.Series, Dict[str, Any]]
    ) -> Tuple[RuleMatch, ...]:
        """
        Evaluate all configured rules against a transaction feature vector.

        Missing Feature Policy:
        - If a target feature is missing from features, is None, or is NaN, the rule condition
          deterministically evaluates to False (no match).
        - Does NOT raise exceptions or crash on missing features.

        Immutability Guarantee:
        - Does NOT mutate the input features mapping in any way.

        Args:
            features: Mapping or pd.Series containing transaction features.

        Returns:
            Tuple[RuleMatch, ...]: Deterministically ordered tuple of triggered rule matches.

        Raises:
            TypeError: If features is not a Mapping or pandas Series.
        """
        if isinstance(features, pd.Series):
            feat_map = features
            is_series = True
        elif isinstance(features, (dict, Mapping)):
            feat_map = features
            is_series = False
        else:
            raise TypeError(
                f"features must be a Mapping, dict, or pandas Series, got {type(features).__name__}"
            )

        matches: List[RuleMatch] = []

        for rule in self._rules:
            # 1. Check feature existence
            if is_series:
                if rule.feature_name not in feat_map.index:
                    continue
                val = feat_map[rule.feature_name]
            else:
                if rule.feature_name not in feat_map:
                    continue
                val = feat_map[rule.feature_name]

            # 2. Handle None or non-finite float
            if val is None:
                continue
            if isinstance(val, (float, np.floating)) and (math.isnan(val) or not math.isfinite(val)):
                continue

            # 3. Evaluate Predicate Operator
            is_matched = False
            op = rule.operator

            if op == RuleOperator.GREATER_THAN:
                if not isinstance(val, bool) and isinstance(val, (int, float, np.number)):
                    is_matched = float(val) > float(rule.comparison_value)
            elif op == RuleOperator.LESS_THAN:
                if not isinstance(val, bool) and isinstance(val, (int, float, np.number)):
                    is_matched = float(val) < float(rule.comparison_value)
            elif op == RuleOperator.IN:
                is_matched = val in rule.comparison_value
            elif op == RuleOperator.IS_TRUE:
                if val is True:
                    is_matched = True
                elif isinstance(val, (int, np.integer)) and not isinstance(val, bool) and val == 1:
                    is_matched = True
                elif isinstance(val, str) and val.strip().lower() in ("true", "1"):
                    is_matched = True

            if is_matched:
                matches.append(
                    RuleMatch(
                        rule_id=rule.rule_id,
                        outcome=rule.outcome,
                        rule_type=rule.rule_type,
                        feature_name=rule.feature_name,
                        feature_value=val,
                        operator=rule.operator,
                        comparison_value=rule.comparison_value,
                        description=rule.description,
                        priority=rule.priority,
                    )
                )

        return tuple(matches)

    def to_dict(self) -> Dict[str, Any]:
        """Convert rule engine configuration to dictionary representation."""
        return {
            "rules": [r.to_dict() for r in self._rules],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleEngine":
        """Reconstitute a RuleEngine instance from dictionary configuration."""
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")
        rule_data = data.get("rules", [])
        if not isinstance(rule_data, list):
            raise TypeError(f"rules must be a list, got {type(rule_data).__name__}")
        return cls([RiskRule.from_dict(r) for r in rule_data])
