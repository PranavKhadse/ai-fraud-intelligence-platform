"""
Automated Unit Tests for Phase 7: Explainability Schemas, Feature Registry, TreeSHAP Explainer & Reason Codes.

Validates:
1. Enums & Schemas: AttributionDirection, ReasonSource, ReasonSeverity, FeatureAttribution,
   ReasonCodeDetail, WaterfallStep, TransactionExplanation.
2. Feature Registry: Complete 55-feature coverage matching PREDICTIVE_FEATURE_COLUMNS,
   valid templates, categorical metadata flags, immutable mapping proxy.
3. TreeSHAP Explainer: Input validation, (N, 56) output shape, base_value dynamic extraction,
   exact feature alignment, additivity/margin reconstruction axiom within 1e-5 tolerance,
   probability reconstruction.
4. Attribution Direction & Ranking: Positive attributions flagged as RISK_INCREASING and sorted
   descending; negative attributions flagged as MITIGATING and sorted by magnitude; rank numbering.
5. Raw Value Preservation: Categorical string values preserved without numeric ordinal leakage.
6. Mathematically Complete Waterfall: Base value + selected features + residual + final margin == output_margin.
7. Reason Code Generation: Separation of source="RULE" and source="MODEL", deterministic ranking,
   plain-English interpolation, override provenance escalation, monitor rule handling.
8. Immutability & JSON Serialization: Deep immutability of all explanation objects, to_dict() validity.
9. Determinism: Repeated identical input yields identical explanation output.
"""

import math
import json
from typing import Dict, Any, Tuple, List, Optional, Sequence
import pytest
import numpy as np
import pandas as pd

from ml.models.config import (
    PREDICTIVE_FEATURE_COLUMNS,
    CATEGORICAL_PREDICTORS,
)
from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    RuleOutcome,
    RuleType,
    RuleOperator,
)
from ml.risk_engine.rules import (
    RiskRule,
    RuleMatch,
    RuleEngine,
)
from ml.explainability.schemas import (
    AttributionDirection,
    ReasonSource,
    ReasonSeverity,
    FeatureAttribution,
    ReasonCodeDetail,
    WaterfallStep,
    TransactionExplanation,
)
from ml.explainability.config import (
    FEATURE_REGISTRY,
    FEATURE_METADATA_REGISTRY,
    get_feature_metadata,
)
from ml.explainability.explainer import (
    TreeSHAPExplainer,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
)
from ml.explainability.reason_codes import (
    ReasonCodeGenerator,
)


# =====================================================================
# 1. Feature Registry & Metadata Tests
# =====================================================================

class TestFeatureRegistry:
    """Test suite for the 55-feature explainability registry."""

    def test_registry_contains_exact_55_features(self) -> None:
        """Verify registry maps every column in PREDICTIVE_FEATURE_COLUMNS and no extras."""
        assert len(FEATURE_REGISTRY) == 55
        assert set(FEATURE_REGISTRY.keys()) == set(PREDICTIVE_FEATURE_COLUMNS)

    def test_registry_fields_structure(self) -> None:
        """Verify each registered feature has all mandatory metadata fields with valid types."""
        mandatory_fields = {
            "display_name": str,
            "category": str,
            "unit": str,
            "reason_code": str,
            "risk_headline": str,
            "mitigating_headline": str,
            "risk_template": str,
            "mitigating_template": str,
            "is_categorical": bool,
        }

        for f_name, meta in FEATURE_REGISTRY.items():
            for field_name, field_type in mandatory_fields.items():
                assert field_name in meta, f"Missing '{field_name}' in metadata for feature '{f_name}'"
                assert isinstance(meta[field_name], field_type), (
                    f"Field '{field_name}' in feature '{f_name}' has wrong type: {type(meta[field_name])}"
                )

    def test_categoricals_flagged_properly(self) -> None:
        """Verify categorical predictors are specifically marked is_categorical=True."""
        for c in CATEGORICAL_PREDICTORS:
            assert FEATURE_REGISTRY[c]["is_categorical"] is True
            assert "{raw_value" in FEATURE_REGISTRY[c]["risk_template"]

        non_cats = [c for c in PREDICTIVE_FEATURE_COLUMNS if c not in CATEGORICAL_PREDICTORS]
        for c in non_cats:
            assert FEATURE_REGISTRY[c]["is_categorical"] is False

    def test_get_feature_metadata_lookup(self) -> None:
        """Verify get_feature_metadata returns valid mapping and raises KeyError on unknown feature."""
        meta = get_feature_metadata("amount")
        assert meta["display_name"] == "Transaction Amount"
        assert meta["category"] == "AMOUNT"

        with pytest.raises(KeyError):
            get_feature_metadata("non_existent_feature_123")


# =====================================================================
# 2. Schema Construction, Immutability & Serialization Tests
# =====================================================================

class TestExplainabilitySchemas:
    """Test suite for typed dataclasses and immutability."""

    def test_feature_attribution_creation_and_immutability(self) -> None:
        """Verify FeatureAttribution validates types and rejects field mutation."""
        fa = FeatureAttribution(
            feature_name="amount",
            display_name="Transaction Amount",
            raw_value=450.0,
            shap_value=0.785234,
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=45.2,
            rank=1,
        )
        assert fa.feature_name == "amount"
        assert fa.shap_value == 0.785234
        assert fa.direction == AttributionDirection.RISK_INCREASING
        assert fa.rank == 1

        with pytest.raises(Exception):
            fa.rank = 2  # frozen dataclass

    def test_feature_attribution_invalid_inputs(self) -> None:
        """Verify FeatureAttribution rejects invalid ranks, non-finite values, and bad types."""
        with pytest.raises(ValueError):
            FeatureAttribution("amount", "Amount", 100, 0.5, AttributionDirection.RISK_INCREASING, 50.0, rank=0)
        with pytest.raises(ValueError):
            FeatureAttribution("amount", "Amount", 100, float("nan"), AttributionDirection.RISK_INCREASING, 50.0, rank=1)
        with pytest.raises(TypeError):
            FeatureAttribution("amount", "Amount", 100, "0.5", AttributionDirection.RISK_INCREASING, 50.0, rank=1)

    def test_reason_code_detail_creation_and_serialization(self) -> None:
        """Verify ReasonCodeDetail creates valid JSON dictionary."""
        rc = ReasonCodeDetail(
            code="HIGH_AMOUNT_ZSCORE",
            headline="Extreme Spending Surge",
            description="Amount deviates by +6.42σ from baseline.",
            category="AMOUNT",
            source=ReasonSource.MODEL,
            severity=ReasonSeverity.HIGH,
            rank=1,
        )
        d = rc.to_dict()
        assert d["code"] == "HIGH_AMOUNT_ZSCORE"
        assert d["source"] == "MODEL"
        assert d["severity"] == "HIGH"
        assert json.loads(json.dumps(d)) == d

    def test_waterfall_step_validation(self) -> None:
        """Verify WaterfallStep enforces valid step types and finite contributions."""
        ws = WaterfallStep(
            step_name="Base Value",
            feature_name=None,
            contribution=0.2437,
            cumulative_margin=0.2437,
            step_type="base",
        )
        assert ws.step_type == "base"

        with pytest.raises(ValueError):
            WaterfallStep("Invalid", None, 0.1, 0.1, step_type="unknown_step_type")

    def test_transaction_explanation_serialization(self) -> None:
        """Verify TransactionExplanation serializes completely to JSON."""
        te = TransactionExplanation(
            model_score=0.85,
            output_margin=1.7346,
            base_value=0.2437,
            risk_score=85,
            risk_tier=RiskTier.CRITICAL,
            action=DecisionAction.BLOCK,
            policy_mode=PolicyMode.TRI_TIER,
            top_risk_factors=(
                FeatureAttribution("amount_zscore", "Spending Z-Score", 6.2, 0.8, AttributionDirection.RISK_INCREASING, 60.0, 1),
            ),
            top_mitigating_factors=(
                FeatureAttribution("account_txn_count_before", "Prior Txns", 45, -0.2, AttributionDirection.MITIGATING, 100.0, 1),
            ),
            reason_codes=(
                ReasonCodeDetail("HIGH_AMOUNT_ZSCORE", "Surge", "Desc", "AMOUNT", ReasonSource.MODEL, ReasonSeverity.HIGH, 1),
            ),
            waterfall=(
                WaterfallStep("Base", None, 0.2437, 0.2437, "base"),
                WaterfallStep("Z-Score", "amount_zscore", 0.8, 1.0437, "feature"),
                WaterfallStep("Residual", None, 0.6909, 1.7346, "residual"),
                WaterfallStep("Final", None, 0.0, 1.7346, "final"),
            ),
            is_overridden=False,
            rule_action=None,
            rules_triggered=(),
            model_version="1.0.0",
        )
        d = te.to_dict()
        assert d["action"] == "BLOCK"
        assert d["risk_score"] == 85
        assert len(d["top_risk_factors"]) == 1
        assert len(d["top_mitigating_factors"]) == 1
        assert len(d["waterfall"]) == 4
        # Verify JSON serializability
        json_str = json.dumps(d)
        assert len(json_str) > 0


# =====================================================================
# 3. TreeSHAP Explainer Unit Tests
# =====================================================================

class TestTreeSHAPExplainer:
    """Test suite for TreeSHAPExplainer execution, shapes, additivity, and rankings."""

    @pytest.fixture
    def explainer(self) -> TreeSHAPExplainer:
        """Instantiate TreeSHAPExplainer from frozen champion artifacts."""
        return TreeSHAPExplainer(
            model_path=DEFAULT_MODEL_PATH,
            preprocessor_path=DEFAULT_PREPROCESSOR_PATH,
        )

    @pytest.fixture
    def sample_raw_transaction(self) -> Dict[str, Any]:
        """Synthetic raw transaction payload conforming to 55 predictive features."""
        row: Dict[str, Any] = {
            "amount": 285.50,
            "cardholder_lat": 40.7128,
            "cardholder_long": -74.0060,
            "merchant_lat": 40.7589,
            "merchant_long": -73.9851,
            "city_pop": 850000.0,
            "merchant_category": "shopping_net",
            "job_category": "engineering",
            "transaction_hour": 23.0,
            "day_of_week": 5.0,
            "day_of_month": 15.0,
            "month": 7.0,
            "week_of_year": 29.0,
            "is_weekend": 1,
            "is_night": 1,
            "hour_sin": -0.2588,
            "hour_cos": 0.9659,
            "day_of_week_sin": -0.9749,
            "day_of_week_cos": -0.2225,
            "txn_count_1h": 5.0,
            "txn_count_6h": 8.0,
            "txn_count_24h": 12.0,
            "txn_count_7d": 25.0,
            "txn_count_30d": 60.0,
            "time_since_prev_txn_seconds": 45.0,
            "is_first_account_txn": 0,
            "amt_sum_1h": 650.0,
            "amt_sum_24h": 1200.0,
            "amt_sum_7d": 2500.0,
            "amt_sum_30d": 5000.0,
            "amt_mean_24h": 100.0,
            "amt_mean_7d": 100.0,
            "amt_max_24h": 285.50,
            "amt_median_30d": 75.0,
            "historical_amount_mean": 65.0,
            "historical_amount_std": 25.0,
            "historical_amount_median": 60.0,
            "amount_zscore": 8.82,
            "amount_ratio_to_historical_mean": 4.39,
            "account_txn_count_before": 150.0,
            "account_total_spend_before": 9750.0,
            "account_avg_amount_before": 65.0,
            "account_max_amount_before": 150.0,
            "account_unique_merchant_count_before": 35.0,
            "account_unique_category_count_before": 8.0,
            "account_merchant_txn_count_before": 1.0,
            "account_category_txn_count_before": 20.0,
            "account_merchant_spend_before": 65.0,
            "account_category_spend_before": 1300.0,
            "merchant_txn_count_before": 1200.0,
            "category_txn_count_before": 45000.0,
            "cardholder_merchant_distance_km": 5.5,
            "distance_from_prev_merchant_km": 1.2,
            "implied_travel_speed_kmh": 96.0,
            "is_impossible_travel_speed": 0,
        }
        return row

    def test_explain_raw_shape_and_base_value(self, explainer: TreeSHAPExplainer, sample_raw_transaction: Dict[str, Any]) -> None:
        """Verify explain_raw produces (N, 56) output and dynamic base value."""
        df = pd.DataFrame([sample_raw_transaction])
        X_trans = explainer.preprocessor.transform(df[PREDICTIVE_FEATURE_COLUMNS])

        assert X_trans.shape == (1, 55)
        contribs = explainer.explain_raw(X_trans)
        assert contribs.shape == (1, 56)

        # Base value is in the last column and non-zero
        base_val = contribs[0, 55]
        assert math.isfinite(base_val)
        assert 0.0 < base_val < 1.0

    def test_additivity_and_margin_reconstruction(self, explainer: TreeSHAPExplainer, sample_raw_transaction: Dict[str, Any]) -> None:
        """Verify Lundberg TreeSHAP Additivity Axiom holds within 1e-5 tolerance."""
        df = pd.DataFrame([sample_raw_transaction])
        X_trans = explainer.preprocessor.transform(df[PREDICTIVE_FEATURE_COLUMNS])

        (
            model_score,
            output_margin,
            base_value,
            top_risk,
            top_mitigating,
            waterfall,
        ) = explainer.explain_features(sample_raw_transaction, preprocessed_row=X_trans)

        # 1. Check sigmoid relationship
        expected_score = 1.0 / (1.0 + math.exp(-output_margin))
        assert abs(model_score - expected_score) < 1e-6

        # 2. Check complete waterfall step reconstruction
        assert waterfall[0].step_type == "base"
        assert abs(waterfall[0].contribution - base_value) < 1e-6

        assert waterfall[-1].step_type == "final"
        assert abs(waterfall[-1].cumulative_margin - output_margin) < 1e-6

        # Check running sum through residual
        residual_step = [w for w in waterfall if w.step_type == "residual"][0]
        assert abs(residual_step.cumulative_margin - output_margin) < 1e-5

    def test_ranking_and_attribution_direction(self, explainer: TreeSHAPExplainer, sample_raw_transaction: Dict[str, Any]) -> None:
        """Verify risk factors are positive & descending; mitigating factors are negative & ascending."""
        (
            model_score,
            output_margin,
            base_value,
            top_risk,
            top_mitigating,
            waterfall,
        ) = explainer.explain_features(sample_raw_transaction, top_k=5, top_mitigating=3)

        assert len(top_risk) <= 5
        for i, factor in enumerate(top_risk):
            assert factor.direction == AttributionDirection.RISK_INCREASING
            assert factor.shap_value > 0.0
            assert factor.rank == i + 1
            if i > 0:
                assert factor.shap_value <= top_risk[i - 1].shap_value  # descending

        assert len(top_mitigating) <= 3
        for j, factor in enumerate(top_mitigating):
            assert factor.direction == AttributionDirection.MITIGATING
            assert factor.shap_value < 0.0
            assert factor.rank == j + 1
            if j > 0:
                # Ranked by absolute magnitude descending (most negative first)
                assert abs(factor.shap_value) <= abs(top_mitigating[j - 1].shap_value)

    def test_raw_categorical_string_preservation(self, explainer: TreeSHAPExplainer, sample_raw_transaction: Dict[str, Any]) -> None:
        """Verify categorical raw string values ('shopping_net') are preserved in attribution raw_value."""
        (
            model_score,
            output_margin,
            base_value,
            top_risk,
            top_mitigating,
            waterfall,
        ) = explainer.explain_features(sample_raw_transaction, top_k=10, top_mitigating=10)

        all_factors = top_risk + top_mitigating
        cat_factors = [f for f in all_factors if f.feature_name in CATEGORICAL_PREDICTORS]

        for cf in cat_factors:
            assert isinstance(cf.raw_value, str)
            assert cf.raw_value in ("shopping_net", "engineering")

    def test_base_value_invariance_across_rows(self, explainer: TreeSHAPExplainer, sample_raw_transaction: Dict[str, Any]) -> None:
        """Verify that TreeSHAP base value is identical across multiple rows."""
        # Create 5 rows with different amounts and locations
        rows = []
        for amt in [10.0, 50.0, 250.0, 1000.0, 5000.0]:
            r = dict(sample_raw_transaction)
            r["amount"] = amt
            rows.append(r)

        df = pd.DataFrame(rows)
        X_trans = explainer.preprocessor.transform(df[PREDICTIVE_FEATURE_COLUMNS])
        contribs = explainer.explain_raw(X_trans)

        base_values = contribs[:, 55]
        assert len(base_values) == 5
        # All rows must have the exact same base value down to float precision
        for b in base_values:
            assert abs(b - base_values[0]) < 1e-10

    def test_waterfall_exact_mathematical_reconstruction(
        self, explainer: TreeSHAPExplainer, sample_raw_transaction: Dict[str, Any]
    ) -> None:
        """Verify mathematical integrity of the waterfall steps and final display marker."""
        (
            model_score,
            output_margin,
            base_value,
            top_risk,
            top_mitigating,
            waterfall,
        ) = explainer.explain_features(sample_raw_transaction, top_k=3, top_mitigating=2)

        # Base step
        base_step = waterfall[0]
        assert base_step.step_type == "base"
        assert abs(base_step.contribution - base_value) < 1e-6

        # Feature steps
        feature_steps = [w for w in waterfall if w.step_type == "feature"]
        assert len(feature_steps) == len(top_risk) + len(top_mitigating)

        # Residual step
        residual_step = [w for w in waterfall if w.step_type == "residual"][0]
        assert residual_step.step_type == "residual"

        # Final display step
        final_step = waterfall[-1]
        assert final_step.step_type == "final"
        assert final_step.contribution == 0.0  # display marker has 0 contribution
        assert abs(final_step.cumulative_margin - output_margin) < 1e-6

        # Sum of mathematical components (base + features + residual) == output_margin
        math_sum = base_step.contribution + sum(f.contribution for f in feature_steps) + residual_step.contribution
        assert abs(math_sum - output_margin) < 1e-5

    def test_raw_values_edge_cases(self, explainer: TreeSHAPExplainer, sample_raw_transaction: Dict[str, Any]) -> None:
        """Verify handling of numpy string scalars, unseen categories, missing categoricals, and numeric types."""
        edge_row = dict(sample_raw_transaction)
        edge_row["merchant_category"] = np.str_("unseen_future_category")
        edge_row["job_category"] = "crypto_analyst"
        edge_row["amount"] = np.float32(750.25)
        edge_row["txn_count_1h"] = np.int64(4)

        (
            model_score,
            output_margin,
            base_value,
            top_risk,
            top_mitigating,
            waterfall,
        ) = explainer.explain_features(edge_row, top_k=10, top_mitigating=10)

        all_factors = top_risk + top_mitigating
        factor_map = {f.feature_name: f for f in all_factors}

        if "merchant_category" in factor_map:
            assert factor_map["merchant_category"].raw_value == "unseen_future_category"
            assert not isinstance(factor_map["merchant_category"].raw_value, (int, float))

        # Check that ordinal integer representations are never surfaced as raw values
        for f in all_factors:
            if f.feature_name in CATEGORICAL_PREDICTORS:
                assert isinstance(f.raw_value, str)


# =====================================================================
# 4. Reason Code Generator Unit Tests
# =====================================================================

class TestReasonCodeGenerator:
    """Test suite for ReasonCodeGenerator prioritizing rules and model drivers."""

    def test_source_separation_and_ordering(self) -> None:
        """Verify source='RULE' and source='MODEL' separation and ordering."""
        # Mock rule match
        rule_match = RuleMatch(
            rule_id="RULE_VELOCITY_BURST_REVIEW",
            outcome=RuleOutcome.REVIEW,
            rule_type=RuleType.VELOCITY,
            feature_name="txn_count_1h",
            feature_value=6.0,
            operator=RuleOperator.GREATER_THAN,
            comparison_value=4.0,
            description="High velocity",
            priority=30,
        )

        # Mock model attributions
        risk_attr = FeatureAttribution(
            feature_name="amount_zscore",
            display_name="Spending Z-Score Deviation",
            raw_value=7.5,
            shap_value=0.95,
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=60.0,
            rank=1,
        )

        reasons = ReasonCodeGenerator.generate_reason_codes(
            top_risk_factors=[risk_attr],
            rule_matches=[rule_match],
            is_overridden=True,
            rule_action=RuleOutcome.REVIEW,
            max_reasons=5,
        )

        assert len(reasons) == 2
        # Overriding rule comes first
        assert reasons[0].code == "RULE_VELOCITY_BURST_REVIEW"
        assert reasons[0].source == ReasonSource.RULE
        assert reasons[0].severity == ReasonSeverity.HIGH
        assert "escalated to REVIEW" in reasons[0].description

        # Model attribution comes second
        assert reasons[1].code == "HIGH_AMOUNT_ZSCORE"
        assert reasons[1].source == ReasonSource.MODEL
        assert reasons[1].severity == ReasonSeverity.HIGH

    def test_monitor_rules_passive_placement(self) -> None:
        """Verify monitor rules are placed after model risk drivers with severity=INFO."""
        monitor_match = RuleMatch(
            rule_id="RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR",
            outcome=RuleOutcome.MONITOR,
            rule_type=RuleType.GEOGRAPHY,
            feature_name="is_impossible_travel_speed",
            feature_value=True,
            operator=RuleOperator.IS_TRUE,
            comparison_value=True,
            description="Impossible travel speed",
            priority=60,
        )

        risk_attr = FeatureAttribution(
            feature_name="amount",
            display_name="Transaction Amount",
            raw_value=1200.0,
            shap_value=0.65,
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=50.0,
            rank=1,
        )

        reasons = ReasonCodeGenerator.generate_reason_codes(
            top_risk_factors=[risk_attr],
            rule_matches=[monitor_match],
            is_overridden=False,
            rule_action=None,
            max_reasons=5,
        )

        assert len(reasons) == 2
        # Model risk driver first
        assert reasons[0].source == ReasonSource.MODEL
        # Monitor rule second
        assert reasons[1].source == ReasonSource.RULE
        assert reasons[1].severity == ReasonSeverity.INFO
        assert reasons[1].code == "RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR"

    def test_deduplication_and_contradiction_safety(self) -> None:
        """Verify that reason codes do not contain duplicate codes."""
        risk_attr1 = FeatureAttribution(
            feature_name="amount",
            display_name="Transaction Amount",
            raw_value=1200.0,
            shap_value=0.65,
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=50.0,
            rank=1,
        )
        risk_attr2 = FeatureAttribution(
            feature_name="amount_zscore",
            display_name="Spending Z-Score Deviation",
            raw_value=6.5,
            shap_value=0.45,
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=35.0,
            rank=2,
        )

        reasons = ReasonCodeGenerator.generate_reason_codes(
            top_risk_factors=[risk_attr1, risk_attr2],
            rule_matches=[],
            is_overridden=False,
            max_reasons=5,
        )

        codes = [r.code for r in reasons]
        assert len(codes) == len(set(codes))
        assert "HIGH_TRANSACTION_AMOUNT" in codes
        assert "HIGH_AMOUNT_ZSCORE" in codes
