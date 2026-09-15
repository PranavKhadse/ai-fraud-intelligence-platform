"""
Unit Tests for Phase 9.6 Increment 1A: Hardened API Persistence Mapping Boundary.

Validates:
1. Low-risk evaluation mapping.
2. Medium-risk evaluation mapping.
3. High-risk / blocked evaluation with business rule override.
4. Transaction identity and external_transaction_id mapping (omitted external ID maps to None).
5. Risk score, tier, decision action, and baseline action mapping.
6. Model metadata provenance.
7. Rule match list mapping.
8. Reason code list mapping.
9. Feature attribution list mapping (risk-increasing and mitigating).
10. Audit log metadata mapping and JSON serializability.
11. Missing account ID rejection (ValueError).
12. Missing model version rejection (ValueError).
13. Currency validation (valid 3-letter ISO code or ValueError on invalid).
14. Invalid enum values rejection (ValueError with allowed options).
15. NaN and Inf rejection across all numeric fields and feature vectors.
16. Numerical out-of-bounds rejection (negative amounts, scores > 1.0, risk_score > 100).
17. Exact 55-feature snapshot keys and missing feature rejection.
18. Timestamp parsing and invalid timestamp rejection.
19. Deterministic transformation output.
20. Invalid input rejection (None inputs).
21. Zero database interaction / pure in-memory transformation.
22. Direct mapping from `TransactionExplanation` domain dataclass.
"""

from datetime import datetime, timezone
from decimal import Decimal
import json
from unittest.mock import patch
import pytest

from backend.app.db.models.enums import (
    AttributionDirection,
    AuditActorType,
    DecisionAction,
    PolicyMode,
    ReasonSeverity,
    ReasonSource,
    RiskTier,
    RuleOutcome,
    RuleType,
)
from backend.app.schemas.predict import (
    FeatureAttributionResponse,
    PredictionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
    TransactionPredictRequest,
)
from backend.app.services.persistence_service import (
    AuditLogData,
    FeatureAttributionData,
    PersistRiskEvaluationCommand,
    ReasonCodeData,
    RiskEvaluationData,
    RuleMatchData,
    TransactionData,
)
from backend.app.services.risk_persistence_mapper import (
    RiskEvaluationContext,
    RiskPersistenceMapper,
    map_explanation_to_command,
    map_prediction_to_command,
)
from ml.explainability.schemas import (
    FeatureAttribution,
    ReasonCodeDetail,
    TransactionExplanation,
    WaterfallStep,
)
from ml.models.config import PREDICTIVE_FEATURE_COLUMNS
from ml.risk_engine.rules import RuleMatch, RuleOperator


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_predict_request() -> TransactionPredictRequest:
    """Canonical 55-feature request fixture."""
    return TransactionPredictRequest(
        transaction_id="TX_REQ_001",
        account_id="ACC_9876",
        timestamp="2026-09-15T12:30:00Z",
        amount=149.99,
        cardholder_lat=40.7128,
        cardholder_long=-74.0060,
        merchant_lat=40.7130,
        merchant_long=-74.0058,
        city_pop=500000.0,
        merchant_category="grocery_pos",
        job_category="engineer",
        transaction_hour=12,
        day_of_week=2,
        day_of_month=15,
        month=9,
        week_of_year=38,
        is_weekend=0,
        is_night=0,
        hour_sin=0.0,
        hour_cos=-1.0,
        day_of_week_sin=0.9749,
        day_of_week_cos=-0.2225,
        txn_count_1h=2.0,
        txn_count_6h=4.0,
        txn_count_24h=7.0,
        txn_count_7d=18.0,
        txn_count_30d=40.0,
        time_since_prev_txn_seconds=7200.0,
        is_first_account_txn=0,
        amt_sum_1h=300.0,
        amt_sum_24h=800.0,
        amt_sum_7d=2500.0,
        amt_sum_30d=6500.0,
        amt_mean_24h=114.28,
        amt_mean_7d=138.88,
        amt_max_24h=250.0,
        amt_median_30d=120.0,
        historical_amount_mean=125.0,
        historical_amount_std=35.0,
        historical_amount_median=120.0,
        amount_zscore=0.714,
        amount_ratio_to_historical_mean=1.20,
        account_txn_count_before=150.0,
        account_total_spend_before=18750.0,
        account_avg_amount_before=125.0,
        account_max_amount_before=450.0,
        account_unique_merchant_count_before=35.0,
        account_unique_category_count_before=10.0,
        account_merchant_txn_count_before=12.0,
        account_category_txn_count_before=28.0,
        account_merchant_spend_before=1400.0,
        account_category_spend_before=3200.0,
        merchant_txn_count_before=2500.0,
        category_txn_count_before=15000.0,
        cardholder_merchant_distance_km=1.85,
        distance_from_prev_merchant_km=0.95,
        implied_travel_speed_kmh=12.5,
        is_impossible_travel_speed=0,
    )


@pytest.fixture
def sample_prediction_response() -> PredictionResponse:
    """Standard PredictionResponse fixture."""
    return PredictionResponse(
        transaction_id="TX_REQ_001",
        model_score=0.150000,
        risk_score=15,
        risk_tier="LOW",
        decision_action="APPROVE",
        policy_mode="TRI_TIER",
        reason="Policy mode TRI_TIER evaluated action APPROVE with normalized risk score 15/100.",
        is_overridden=False,
        rule_action=None,
        rules_triggered=[],
        rule_matches=[],
        reason_codes=[
            ReasonCodeResponse(
                code="RC_NORMAL_AMOUNT",
                headline="Standard Transaction Amount",
                description="Transaction amount is within typical historical range",
                category="AMOUNT",
                source="MODEL",
                severity="INFO",
                rank=1,
            )
        ],
        top_risk_factors=[
            FeatureAttributionResponse(
                feature_name="amount",
                display_name="Transaction Amount",
                raw_value=149.99,
                shap_value=0.120000,
                direction="RISK_INCREASING",
                relative_contribution_pct=0.7500,
                rank=1,
            )
        ],
        top_mitigating_factors=[
            FeatureAttributionResponse(
                feature_name="account_txn_count_before",
                display_name="Account Transaction History",
                raw_value=150.0,
                shap_value=-0.350000,
                direction="MITIGATING",
                relative_contribution_pct=1.0000,
                rank=1,
            )
        ],
        model_version="1.0.0",
        evaluated_at="2026-09-15T12:30:05.123456+00:00",
    )


# ==============================================================================
# Unit Test Cases
# ==============================================================================

class TestRiskPersistenceMapper:
    """Comprehensive test suite for the hardened API-to-persistence mapper."""

    def test_1_map_low_risk_prediction(self, sample_predict_request, sample_prediction_response):
        """Test 1: Low-risk APPROVE prediction mapping."""
        cmd = RiskPersistenceMapper.map_prediction_to_command(
            request=sample_predict_request,
            response=sample_prediction_response,
        )

        assert isinstance(cmd, PersistRiskEvaluationCommand)
        assert cmd.transaction.amount == Decimal("149.99")
        assert cmd.transaction.external_transaction_id == "TX_REQ_001"
        assert cmd.transaction.account_id == "ACC_9876"
        assert cmd.evaluation.model_score == Decimal("0.150000")
        assert cmd.evaluation.risk_score == 15
        assert cmd.evaluation.risk_tier == RiskTier.LOW
        assert cmd.evaluation.decision_action == DecisionAction.APPROVE
        assert cmd.evaluation.baseline_action == DecisionAction.APPROVE
        assert cmd.evaluation.is_overridden is False
        assert len(cmd.reason_codes) == 1
        assert len(cmd.feature_attributions) == 2
        assert len(cmd.rule_matches) == 0

    def test_2_map_medium_risk_prediction(self, sample_predict_request):
        """Test 2: Medium-risk REVIEW prediction mapping."""
        resp = PredictionResponse(
            transaction_id="TX_REQ_002",
            model_score=0.485000,
            risk_score=49,
            risk_tier="MEDIUM",
            decision_action="REVIEW",
            policy_mode="TRI_TIER",
            reason="Policy mode TRI_TIER evaluated action REVIEW with normalized risk score 49/100.",
            is_overridden=False,
            rule_action=None,
            rules_triggered=[],
            rule_matches=[],
            reason_codes=[],
            top_risk_factors=[],
            top_mitigating_factors=[],
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:31:00Z",
        )

        cmd = RiskPersistenceMapper.map_prediction_to_command(
            request=sample_predict_request,
            response=resp,
        )

        assert cmd.evaluation.risk_tier == RiskTier.MEDIUM
        assert cmd.evaluation.decision_action == DecisionAction.REVIEW
        assert cmd.evaluation.model_score == Decimal("0.485000")
        assert cmd.evaluation.risk_score == 49

    def test_3_map_high_risk_rule_overridden_prediction(self, sample_predict_request):
        """Test 3: High-risk BLOCK prediction with rule override."""
        resp = PredictionResponse(
            transaction_id="TX_REQ_003",
            model_score=0.320000,
            risk_score=32,
            risk_tier="HIGH",
            decision_action="BLOCK",
            policy_mode="TRI_TIER",
            reason="Baseline ML action overridden to BLOCK by rule(s): RULE_VELOCITY_BURST.",
            is_overridden=True,
            rule_action="BLOCK",
            rules_triggered=["RULE_VELOCITY_BURST"],
            rule_matches=[
                RuleMatchResponse(
                    rule_id="RULE_VELOCITY_BURST",
                    description="Excessive transaction velocity",
                    feature_name="txn_count_1h",
                    operator=">",
                    comparison_value="5.0",
                    outcome="BLOCK",
                    rule_type="VELOCITY",
                    priority=5,
                )
            ],
            reason_codes=[
                ReasonCodeResponse(
                    code="RC_VELOCITY_BURST",
                    headline="Rapid Velocity Trigger",
                    description="Triggered critical velocity rule",
                    category="VELOCITY",
                    source="RULE",
                    severity="CRITICAL",
                    rank=1,
                )
            ],
            top_risk_factors=[],
            top_mitigating_factors=[],
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:32:00Z",
        )

        cmd = RiskPersistenceMapper.map_prediction_to_command(
            request=sample_predict_request,
            response=resp,
        )

        assert cmd.evaluation.is_overridden is True
        assert cmd.evaluation.decision_action == DecisionAction.BLOCK
        assert cmd.evaluation.rule_action == RuleOutcome.BLOCK
        assert len(cmd.rule_matches) == 1
        assert cmd.rule_matches[0].rule_id == "RULE_VELOCITY_BURST"
        assert cmd.rule_matches[0].outcome == RuleOutcome.BLOCK
        assert cmd.rule_matches[0].rule_type == RuleType.VELOCITY
        assert cmd.rule_matches[0].priority == 5

    def test_4_map_transaction_identity_and_external_id(self, sample_predict_request, sample_prediction_response):
        """Test 4: Verify external transaction ID extraction and fallback handling."""
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert cmd.transaction.external_transaction_id == "TX_REQ_001"

        # Explicit context override
        ctx = RiskEvaluationContext(external_transaction_id="TX_OVERRIDE_999")
        cmd2 = map_prediction_to_command(sample_predict_request, sample_prediction_response, context=ctx)
        assert cmd2.transaction.external_transaction_id == "TX_OVERRIDE_999"

    def test_5_map_risk_score_and_actions(self, sample_predict_request, sample_prediction_response):
        """Test 5: Verify risk score, tier, decision action, and baseline action mapping."""
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert cmd.evaluation.risk_score == 15
        assert cmd.evaluation.risk_tier == RiskTier.LOW
        assert cmd.evaluation.decision_action == DecisionAction.APPROVE
        assert cmd.evaluation.baseline_action == DecisionAction.APPROVE

    def test_6_map_model_metadata(self, sample_predict_request, sample_prediction_response):
        """Test 6: Verify model version and policy mode provenance."""
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert cmd.evaluation.model_version == "1.0.0"
        assert cmd.evaluation.policy_mode == PolicyMode.TRI_TIER

    def test_7_map_rule_matches(self, sample_predict_request):
        """Test 7: Verify RuleMatchData list conversion."""
        resp = PredictionResponse(
            transaction_id="TX_REQ_007",
            model_score=0.850000,
            risk_score=85,
            risk_tier="HIGH",
            decision_action="BLOCK",
            policy_mode="TRI_TIER",
            reason="Blocked by policy",
            is_overridden=False,
            rule_matches=[
                RuleMatchResponse(
                    rule_id="RULE_GEO_SPEED",
                    description="Impossible speed",
                    feature_name="implied_travel_speed_kmh",
                    operator=">",
                    comparison_value=800.0,
                    outcome="BLOCK",
                    rule_type="GEOGRAPHY",
                    priority=1,
                )
            ],
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:35:00Z",
        )
        cmd = map_prediction_to_command(sample_predict_request, resp)
        assert len(cmd.rule_matches) == 1
        assert cmd.rule_matches[0].rule_id == "RULE_GEO_SPEED"
        assert cmd.rule_matches[0].operator == ">"
        assert cmd.rule_matches[0].comparison_value == "800.0"
        assert cmd.rule_matches[0].outcome == RuleOutcome.BLOCK
        assert cmd.rule_matches[0].rule_type == RuleType.GEOGRAPHY
        assert cmd.rule_matches[0].priority == 1

    def test_8_map_reason_codes(self, sample_predict_request, sample_prediction_response):
        """Test 8: Verify ReasonCodeData list conversion."""
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert len(cmd.reason_codes) == 1
        rc = cmd.reason_codes[0]
        assert rc.code == "RC_NORMAL_AMOUNT"
        assert rc.headline == "Standard Transaction Amount"
        assert rc.category == "AMOUNT"
        assert rc.source == ReasonSource.MODEL
        assert rc.severity == ReasonSeverity.INFO
        assert rc.rank == 1

    def test_9_map_feature_attributions(self, sample_predict_request, sample_prediction_response):
        """Test 9: Verify feature attribution mapping for both risk and mitigating factors."""
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert len(cmd.feature_attributions) == 2

        risk_fa = cmd.feature_attributions[0]
        assert risk_fa.feature_name == "amount"
        assert risk_fa.shap_value == Decimal("0.120000")
        assert risk_fa.direction == AttributionDirection.RISK_INCREASING
        assert risk_fa.relative_contribution_pct == Decimal("0.7500")

        mit_fa = cmd.feature_attributions[1]
        assert mit_fa.feature_name == "account_txn_count_before"
        assert mit_fa.shap_value == Decimal("-0.350000")
        assert mit_fa.direction == AttributionDirection.MITIGATING
        assert mit_fa.relative_contribution_pct == Decimal("1.0000")

    def test_10_map_audit_data(self, sample_predict_request, sample_prediction_response):
        """Test 10: Verify AuditLogData conversion with HTTP context."""
        ctx = RiskEvaluationContext(
            correlation_id="corr-xyz-123",
            client_ip="192.168.1.50",
            actor_id="test_api_client",
            actor_type=AuditActorType.API_CLIENT,
            evaluation_latency_ms=12.55,
        )
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response, context=ctx)
        assert cmd.audit is not None
        assert cmd.audit.correlation_id == "corr-xyz-123"
        assert cmd.audit.client_ip == "192.168.1.50"
        assert cmd.audit.actor_id == "test_api_client"
        assert cmd.audit.actor_type == AuditActorType.API_CLIENT
        assert cmd.audit.payload["external_transaction_id"] == "TX_REQ_001"
        assert cmd.audit.payload["model_score"] == 0.15
        assert cmd.evaluation.evaluation_latency_ms == Decimal("12.55")

    def test_11_optional_external_transaction_id_none_handling(self, sample_predict_request, sample_prediction_response):
        """Test 11: Verify omitted external transaction ID is preserved as None."""
        req_dict = sample_predict_request.model_dump()
        req_dict["transaction_id"] = None

        resp = PredictionResponse(
            transaction_id=None,
            model_score=0.100000,
            risk_score=10,
            risk_tier="LOW",
            decision_action="APPROVE",
            policy_mode="TRI_TIER",
            reason="Low risk",
            is_overridden=False,
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:00:00Z",
        )

        cmd = map_prediction_to_command(req_dict, resp)
        assert cmd.transaction.external_transaction_id is None
        assert cmd.transaction.account_id == "ACC_9876"

    def test_12_missing_account_id_raises_value_error(self, sample_predict_request, sample_prediction_response):
        """Test 12: Missing or empty account_id in both request and context raises ValueError."""
        req_dict = sample_predict_request.model_dump()
        req_dict["account_id"] = None

        with pytest.raises(ValueError, match="Missing required 'account_id'"):
            map_prediction_to_command(req_dict, sample_prediction_response)

        # But if account_id is supplied via context, it succeeds
        ctx = RiskEvaluationContext(account_id="ACC_FROM_CONTEXT")
        cmd = map_prediction_to_command(req_dict, sample_prediction_response, context=ctx)
        assert cmd.transaction.account_id == "ACC_FROM_CONTEXT"

    def test_13_missing_or_empty_model_version_raises_value_error(self, sample_predict_request):
        """Test 13: Missing model_version in PredictionResponse raises ValueError."""
        resp = PredictionResponse(
            transaction_id="TX_123",
            model_score=0.2,
            risk_score=20,
            risk_tier="LOW",
            decision_action="APPROVE",
            policy_mode="TRI_TIER",
            reason="Test",
            is_overridden=False,
            model_version=None,
            evaluated_at="2026-09-15T12:00:00Z",
        )
        with pytest.raises(ValueError, match="model_version is required"):
            map_prediction_to_command(sample_predict_request, resp)

    def test_14_invalid_currency_code_raises_value_error(self, sample_predict_request, sample_prediction_response):
        """Test 14: Invalid currency format raises ValueError."""
        ctx = RiskEvaluationContext(currency="TOOLONG")
        with pytest.raises(ValueError, match="Invalid currency code"):
            map_prediction_to_command(sample_predict_request, sample_prediction_response, context=ctx)

    def test_15_invalid_enum_values_raise_value_error(self, sample_predict_request):
        """Test 15: Invalid string enum inputs raise ValueError with allowed list."""
        resp = PredictionResponse(
            transaction_id="TX_123",
            model_score=0.2,
            risk_score=20,
            risk_tier="INVALID_TIER",
            decision_action="APPROVE",
            policy_mode="TRI_TIER",
            reason="Test",
            is_overridden=False,
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:00:00Z",
        )
        with pytest.raises(ValueError, match="Invalid value 'INVALID_TIER' for enum RiskTier"):
            map_prediction_to_command(sample_predict_request, resp)

    def test_16_nan_and_inf_numeric_values_raise_value_error(self, sample_predict_request, sample_prediction_response):
        """Test 16: NaN and Infinite values raise ValueError."""
        # 1. NaN in amount
        req_dict = sample_predict_request.model_dump()
        req_dict["amount"] = float("nan")
        with pytest.raises(ValueError, match="must be finite"):
            map_prediction_to_command(req_dict, sample_prediction_response)

        # 2. Inf in model_score
        resp_inf = PredictionResponse(
            transaction_id="TX_123",
            model_score=float("inf"),
            risk_score=50,
            risk_tier="MEDIUM",
            decision_action="REVIEW",
            policy_mode="TRI_TIER",
            reason="Test",
            is_overridden=False,
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:00:00Z",
        )
        with pytest.raises(ValueError, match="must be finite"):
            map_prediction_to_command(sample_predict_request, resp_inf)

    def test_17_out_of_bounds_numerical_values_raise_value_error(self, sample_predict_request):
        """Test 17: Out of bounds numerical values raise ValueError."""
        # Negative amount
        req_dict = sample_predict_request.model_dump()
        req_dict["amount"] = -10.0
        with pytest.raises(ValueError, match="amount must be non-negative"):
            map_prediction_to_command(req_dict, sample_prediction_response)

        # Score > 1.0
        resp_score = PredictionResponse(
            transaction_id="TX_123",
            model_score=1.5,
            risk_score=50,
            risk_tier="MEDIUM",
            decision_action="REVIEW",
            policy_mode="TRI_TIER",
            reason="Test",
            is_overridden=False,
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:00:00Z",
        )
        with pytest.raises(ValueError, match="model_score must be in"):
            map_prediction_to_command(sample_predict_request, resp_score)

        # Risk score > 100
        resp_risk = PredictionResponse(
            transaction_id="TX_123",
            model_score=0.5,
            risk_score=150,
            risk_tier="MEDIUM",
            decision_action="REVIEW",
            policy_mode="TRI_TIER",
            reason="Test",
            is_overridden=False,
            model_version="1.0.0",
            evaluated_at="2026-09-15T12:00:00Z",
        )
        with pytest.raises(ValueError, match="risk_score must be an integer in"):
            map_prediction_to_command(sample_predict_request, resp_risk)

    def test_18_feature_snapshot_exact_55_keys_and_missing_feature_rejection(self, sample_predict_request, sample_prediction_response):
        """Test 18: Verify features_snapshot contains exactly 55 predictive features and rejects missing keys."""
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert len(cmd.transaction.features_snapshot) == 55
        assert set(cmd.transaction.features_snapshot.keys()) == set(PREDICTIVE_FEATURE_COLUMNS)

        # Missing a predictive feature raises ValueError
        sparse_dict = sample_predict_request.model_dump()
        del sparse_dict["amt_sum_1h"]
        with pytest.raises(ValueError, match="Missing 1 required predictive feature column"):
            map_prediction_to_command(sparse_dict, sample_prediction_response)

    def test_19_timestamp_parsing_and_invalid_timestamp_error(self, sample_predict_request, sample_prediction_response):
        """Test 19: Verify valid timestamps parse to UTC and invalid format raises ValueError."""
        req_dict = sample_predict_request.model_dump()
        req_dict["timestamp"] = "not-a-valid-datetime"
        with pytest.raises(ValueError, match="Invalid datetime format"):
            map_prediction_to_command(req_dict, sample_prediction_response)

    def test_20_deterministic_output(self, sample_predict_request, sample_prediction_response):
        """Test 20: Verify repeated mapping on identical inputs produces identical outputs."""
        cmd1 = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        cmd2 = map_prediction_to_command(sample_predict_request, sample_prediction_response)

        assert cmd1.transaction.amount == cmd2.transaction.amount
        assert cmd1.evaluation.model_score == cmd2.evaluation.model_score
        assert cmd1.evaluation.risk_score == cmd2.evaluation.risk_score
        assert len(cmd1.feature_attributions) == len(cmd2.feature_attributions)

    def test_21_invalid_input_rejection(self, sample_predict_request, sample_prediction_response):
        """Test 21: Verify ValueError when request or response is None."""
        with pytest.raises(ValueError, match="Request payload must not be None"):
            RiskPersistenceMapper.map_prediction_to_command(None, sample_prediction_response)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="PredictionResponse must not be None"):
            RiskPersistenceMapper.map_prediction_to_command(sample_predict_request, None)  # type: ignore[arg-type]

    def test_22_zero_database_access_confirmation(self, sample_predict_request, sample_prediction_response):
        """Test 22: Confirm mapper executes zero database connections, engines, or sessions."""
        with patch("backend.app.db.session.get_async_engine") as mock_engine, \
             patch("backend.app.db.session.get_db_session") as mock_session:
            cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
            assert cmd is not None
            mock_engine.assert_not_called()
            mock_session.assert_not_called()

    def test_23_map_from_transaction_explanation_domain_object(self, sample_predict_request):
        """Test 23: Verify direct mapping from TransactionExplanation dataclass."""
        explanation = TransactionExplanation(
            model_score=0.725000,
            output_margin=0.969400,
            base_value=-1.500000,
            risk_score=73,
            risk_tier=RiskTier.HIGH,
            action=DecisionAction.REVIEW,
            baseline_action=DecisionAction.REVIEW,
            policy_mode=PolicyMode.TRI_TIER,
            top_risk_factors=(
                FeatureAttribution(
                    feature_name="amount_zscore",
                    display_name="Amount Z-Score",
                    raw_value=2.55,
                    shap_value=0.550000,
                    direction=AttributionDirection.RISK_INCREASING,
                    relative_contribution_pct=0.6500,
                    rank=1,
                ),
            ),
            top_mitigating_factors=(
                FeatureAttribution(
                    feature_name="time_since_prev_txn_seconds",
                    display_name="Time Since Previous Transaction",
                    raw_value=3600.0,
                    shap_value=-0.200000,
                    direction=AttributionDirection.MITIGATING,
                    relative_contribution_pct=0.3500,
                    rank=1,
                ),
            ),
            reason_codes=(
                ReasonCodeDetail(
                    code="RC_AMOUNT_ZSCORE",
                    headline="Unusual Spending Deviation",
                    description="Transaction amount deviates from historical profile",
                    category="AMOUNT",
                    source=ReasonSource.MODEL,
                    severity=ReasonSeverity.HIGH,
                    rank=1,
                ),
            ),
            waterfall=(
                WaterfallStep(
                    step_name="Base Value",
                    feature_name=None,
                    contribution=-1.5,
                    cumulative_margin=-1.5,
                    step_type="base",
                ),
            ),
            is_overridden=False,
            rule_action=None,
            rules_triggered=(),
            rule_matches=(
                RuleMatch(
                    rule_id="RULE_VELOCITY_REVIEW",
                    description="Velocity review",
                    feature_name="txn_count_1h",
                    feature_value=5.0,
                    operator=RuleOperator.GREATER_THAN,
                    comparison_value=4.0,
                    outcome=RuleOutcome.REVIEW,
                    rule_type=RuleType.VELOCITY,
                    priority=10,
                ),
            ),
            model_version="1.0.0",
        )

        ctx = RiskEvaluationContext(
            correlation_id="corr-expl-123",
            evaluation_latency_ms=14.8,
        )

        cmd = map_explanation_to_command(sample_predict_request, explanation, context=ctx)

        assert isinstance(cmd, PersistRiskEvaluationCommand)
        assert cmd.evaluation.model_score == Decimal("0.725000")
        assert cmd.evaluation.output_margin == Decimal("0.969400")
        assert cmd.evaluation.base_value == Decimal("-1.500000")
        assert cmd.evaluation.risk_tier == RiskTier.HIGH
        assert cmd.evaluation.decision_action == DecisionAction.REVIEW
        assert cmd.evaluation.baseline_action == DecisionAction.REVIEW
        assert len(cmd.rule_matches) == 1
        assert cmd.rule_matches[0].rule_id == "RULE_VELOCITY_REVIEW"
        assert len(cmd.reason_codes) == 1
        assert len(cmd.feature_attributions) == 2
        assert cmd.evaluation.correlation_id == "corr-expl-123"
        assert cmd.evaluation.evaluation_latency_ms == Decimal("14.80")

    def test_24_audit_payload_json_serializability(self, sample_predict_request, sample_prediction_response):
        """Test 24: Verify AuditLogData payload is strictly JSON-serializable."""
        cmd = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert cmd.audit is not None
        # Must serialize to JSON without throwing TypeError
        serialized = json.dumps(cmd.audit.payload)
        assert "TX_REQ_001" in serialized
        assert "APPROVE" in serialized

    def test_25_timestamp_distinction_and_fallback(self, sample_predict_request, sample_prediction_response):
        """Test 25: Verify explicit transaction timestamp vs ingestion fallback vs evaluated_at distinction."""
        # 1. Explicit timestamp in request is preserved
        cmd_explicit = map_prediction_to_command(sample_predict_request, sample_prediction_response)
        assert cmd_explicit.transaction.transaction_timestamp == datetime(2026, 9, 15, 12, 30, 0, tzinfo=timezone.utc)
        assert cmd_explicit.evaluation.evaluated_at == datetime(2026, 9, 15, 12, 30, 5, 123456, tzinfo=timezone.utc)

        # 2. Context override for transaction timestamp
        ctx = RiskEvaluationContext(transaction_timestamp="2026-08-01T08:00:00Z")
        cmd_ctx = map_prediction_to_command(sample_predict_request, sample_prediction_response, context=ctx)
        assert cmd_ctx.transaction.transaction_timestamp == datetime(2026, 8, 1, 8, 0, 0, tzinfo=timezone.utc)

        # 3. Omitted timestamp uses explicit UTC ingestion fallback
        req_no_ts = sample_predict_request.model_dump()
        req_no_ts["timestamp"] = None
        before = datetime.now(timezone.utc)
        cmd_fallback = map_prediction_to_command(req_no_ts, sample_prediction_response)
        after = datetime.now(timezone.utc)
        assert before <= cmd_fallback.transaction.transaction_timestamp <= after

    def test_26_empty_string_timestamp_uses_fallback(self, sample_predict_request, sample_prediction_response):
        """Test 26: Empty/whitespace timestamp string triggers ingestion fallback."""
        req_blank_ts = sample_predict_request.model_dump()
        req_blank_ts["timestamp"] = "   "
        before = datetime.now(timezone.utc)
        cmd = map_prediction_to_command(req_blank_ts, sample_prediction_response)
        after = datetime.now(timezone.utc)
        assert before <= cmd.transaction.transaction_timestamp <= after

