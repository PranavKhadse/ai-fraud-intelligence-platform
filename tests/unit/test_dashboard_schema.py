"""
Unit Tests for Dashboard Pydantic Schemas (Increment 11.1).

Tests:
1. Valid serialization of DashboardOverviewResponse.
2. Boundary constraints (rates and risk score between 0.0 and 100.0, counts >= 0, amount >= 0.0).
3. Negative values rejection.
4. Extra fields rejection (ConfigDict(extra="forbid")).
5. Safe zero-value defaults via DashboardOverviewResponse.empty().
"""

import pytest
from pydantic import ValidationError

from backend.app.schemas.dashboard import DashboardOverviewResponse


class TestDashboardOverviewSchema:
    """Test suite for DashboardOverviewResponse Pydantic model."""

    def test_valid_dashboard_overview_response(self) -> None:
        """Verify successful instantiation with typical valid production payload."""
        data = {
            "total_transactions": 10000,
            "total_amount": 542980.50,
            "approval_count": 9300,
            "approval_rate": 93.00,
            "review_count": 450,
            "review_rate": 4.50,
            "block_count": 250,
            "block_rate": 2.50,
            "average_risk_score": 18.45,
            "average_latency_ms": 24.85,
        }
        resp = DashboardOverviewResponse(**data)
        assert resp.total_transactions == 10000
        assert resp.total_amount == 542980.50
        assert resp.approval_count == 9300
        assert resp.approval_rate == 93.00
        assert resp.review_count == 450
        assert resp.review_rate == 4.50
        assert resp.block_count == 250
        assert resp.block_rate == 2.50
        assert resp.average_risk_score == 18.45
        assert resp.average_latency_ms == 24.85

    def test_empty_factory_method(self) -> None:
        """Verify the empty() classmethod produces compliant zero-state metrics."""
        empty_resp = DashboardOverviewResponse.empty()
        assert empty_resp.total_transactions == 0
        assert empty_resp.total_amount == 0.0
        assert empty_resp.approval_count == 0
        assert empty_resp.approval_rate == 0.0
        assert empty_resp.review_count == 0
        assert empty_resp.review_rate == 0.0
        assert empty_resp.block_count == 0
        assert empty_resp.block_rate == 0.0
        assert empty_resp.average_risk_score == 0.0
        assert empty_resp.average_latency_ms == 0.0

    @pytest.mark.parametrize(
        "field,invalid_value",
        [
            ("total_transactions", -1),
            ("total_amount", -10.5),
            ("approval_count", -5),
            ("approval_rate", -0.1),
            ("approval_rate", 100.1),
            ("review_count", -1),
            ("review_rate", -1.0),
            ("review_rate", 105.0),
            ("block_count", -2),
            ("block_rate", -0.01),
            ("block_rate", 100.01),
            ("average_risk_score", -0.5),
            ("average_risk_score", 100.5),
            ("average_latency_ms", -1.0),
        ],
    )
    def test_boundary_validation_failures(self, field: str, invalid_value: float) -> None:
        """Verify boundary violations trigger ValidationError."""
        valid_data = {
            "total_transactions": 100,
            "total_amount": 1000.0,
            "approval_count": 90,
            "approval_rate": 90.0,
            "review_count": 7,
            "review_rate": 7.0,
            "block_count": 3,
            "block_rate": 3.0,
            "average_risk_score": 25.0,
            "average_latency_ms": 15.0,
        }
        valid_data[field] = invalid_value
        with pytest.raises(ValidationError):
            DashboardOverviewResponse(**valid_data)

    def test_forbid_extra_fields(self) -> None:
        """Verify arbitrary extraneous fields are rejected."""
        data = {
            "total_transactions": 0,
            "total_amount": 0.0,
            "approval_count": 0,
            "approval_rate": 0.0,
            "review_count": 0,
            "review_rate": 0.0,
            "block_count": 0,
            "block_rate": 0.0,
            "average_risk_score": 0.0,
            "average_latency_ms": 0.0,
            "unauthorized_secret": "injected_data",
        }
        with pytest.raises(ValidationError):
            DashboardOverviewResponse(**data)


from datetime import datetime, timezone
from backend.app.db.models.enums import (
    AttributionDirection,
    DecisionAction,
    PolicyMode,
    ReasonSeverity,
    ReasonSource,
    RiskTier,
    RuleOutcome,
    RuleType,
)
from backend.app.schemas.dashboard import (
    AuditLogDetail,
    EvaluationDetail,
    FeatureAttributionDetail,
    ReasonCodeDetail,
    RuleMatchDetail,
    TransactionDetailResponse,
    TransactionListItem,
    TransactionListResponse,
    TransactionMetadataDetail,
)


class TestTransactionListSchemas:
    """Test suite for TransactionListItem and TransactionListResponse schemas."""

    def test_valid_transaction_list_item(self) -> None:
        """Verify instantiation of TransactionListItem with complete valid data."""
        item = TransactionListItem(
            id="c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f",
            external_transaction_id="TX_12345",
            transaction_timestamp=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
            amount=299.50,
            currency="USD",
            merchant_category="electronics",
            risk_score=75,
            risk_tier=RiskTier.HIGH,
            decision_action=DecisionAction.REVIEW,
            is_overridden=True,
            evaluation_latency_ms=12.4,
        )
        assert item.id == "c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f"
        assert item.external_transaction_id == "TX_12345"
        assert item.amount == 299.50
        assert item.currency == "USD"
        assert item.merchant_category == "electronics"
        assert item.risk_score == 75
        assert item.risk_tier == RiskTier.HIGH
        assert item.decision_action == DecisionAction.REVIEW
        assert item.is_overridden is True
        assert item.evaluation_latency_ms == 12.4

    def test_transaction_list_item_boundary_failures(self) -> None:
        """Verify invalid values in TransactionListItem raise ValidationError."""
        base_data = {
            "id": "c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f",
            "transaction_timestamp": "2026-09-17T12:00:00Z",
            "amount": 100.0,
            "currency": "USD",
            "merchant_category": "retail",
            "risk_score": 50,
            "risk_tier": RiskTier.MEDIUM,
            "decision_action": DecisionAction.APPROVE,
            "is_overridden": False,
        }

        # Negative amount
        with pytest.raises(ValidationError):
            TransactionListItem(**{**base_data, "amount": -10.0})

        # Score > 100
        with pytest.raises(ValidationError):
            TransactionListItem(**{**base_data, "risk_score": 101})

        # Score < 0
        with pytest.raises(ValidationError):
            TransactionListItem(**{**base_data, "risk_score": -1})

        # Negative latency
        with pytest.raises(ValidationError):
            TransactionListItem(**{**base_data, "evaluation_latency_ms": -5.0})

        # Extra forbid field
        with pytest.raises(ValidationError):
            TransactionListItem(**{**base_data, "secret_features": [1, 2, 3]})

    def test_valid_transaction_list_response(self) -> None:
        """Verify TransactionListResponse structure and pagination boundaries."""
        item = TransactionListItem(
            id="c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f",
            transaction_timestamp=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
            amount=50.0,
            merchant_category="grocery_pos",
            risk_score=10,
            risk_tier=RiskTier.LOW,
            decision_action=DecisionAction.APPROVE,
        )
        resp = TransactionListResponse(
            items=[item],
            total_count=1,
            limit=20,
            offset=0,
        )
        assert len(resp.items) == 1
        assert resp.total_count == 1
        assert resp.limit == 20
        assert resp.offset == 0

    def test_transaction_list_response_boundaries(self) -> None:
        """Verify pagination boundary validation in TransactionListResponse."""
        # Limit > 100
        with pytest.raises(ValidationError):
            TransactionListResponse(items=[], total_count=0, limit=101, offset=0)

        # Limit < 1
        with pytest.raises(ValidationError):
            TransactionListResponse(items=[], total_count=0, limit=0, offset=0)

        # Offset < 0
        with pytest.raises(ValidationError):
            TransactionListResponse(items=[], total_count=0, limit=20, offset=-1)

        # Total count < 0
        with pytest.raises(ValidationError):
            TransactionListResponse(items=[], total_count=-1, limit=20, offset=0)


class TestTransactionDetailSchemas:
    """Test suite for TransactionDetailResponse and nested investigation schemas (Increment 11.3)."""

    def test_valid_transaction_detail_response(self) -> None:
        """Verify instantiation of TransactionDetailResponse with complete nested structure."""
        now = datetime(2026, 9, 17, 14, 30, 0, tzinfo=timezone.utc)
        meta = TransactionMetadataDetail(
            id="c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f",
            external_transaction_id="TX_987654",
            account_id="ACC_102938",
            merchant_id="MERCH_5544",
            merchant_category="electronics",
            job_category="engineering",
            amount=499.99,
            currency="USD",
            cardholder_lat=37.7749,
            cardholder_long=-122.4194,
            merchant_lat=37.7833,
            merchant_long=-122.4167,
            city_pop=873965,
            transaction_timestamp=now,
            created_at=now,
        )

        eval_detail = EvaluationDetail(
            id="e4f5a6b7-c8d9-0e1f-2a3b-4c5d6e7f8a9b",
            model_version="1.0.0",
            policy_mode=PolicyMode.TRI_TIER,
            model_score=0.8452,
            risk_score=85,
            risk_tier=RiskTier.HIGH,
            decision_action=DecisionAction.BLOCK,
            baseline_action=DecisionAction.REVIEW,
            is_overridden=True,
            rule_action=RuleOutcome.BLOCK,
            decision_reason="Risk score 85 meets BLOCK threshold.",
            output_margin=1.725,
            base_value=-3.542,
            evaluation_latency_ms=14.8,
            correlation_id="corr_12345",
            evaluated_at=now,
        )

        fa = FeatureAttributionDetail(
            id="f1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d",
            feature_name="txn_count_1h",
            display_name="Transactions in Past 1 Hour",
            raw_value=6,
            shap_value=1.452,
            direction=AttributionDirection.RISK_INCREASING,
            relative_contribution_pct=42.5,
            rank=1,
        )

        rc = ReasonCodeDetail(
            id="r1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d",
            code="VELOCITY_BURST_1H",
            headline="Rapid Transaction Velocity",
            description="6 transactions attempted within the last hour.",
            category="VELOCITY",
            source=ReasonSource.MODEL,
            severity=ReasonSeverity.HIGH,
            rank=1,
        )

        rm = RuleMatchDetail(
            id="m1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d",
            rule_id="RULE_VELOCITY_BURST_BLOCK",
            description="Block cardholder when hourly transaction count exceeds 5.",
            feature_name="txn_count_1h",
            operator=">",
            comparison_value="5",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.VELOCITY,
            priority=10,
        )

        audit = AuditLogDetail(
            id="a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
            event_type="RISK_EVALUATION_PERSISTED",
            action="PERSIST_EVALUATION",
            actor_type="SYSTEM",
            actor_id="fraud_engine_worker_01",
            correlation_id="corr_12345",
            client_ip="192.168.1.100",
            event_timestamp=now,
        )

        resp = TransactionDetailResponse(
            transaction=meta,
            evaluation=eval_detail,
            features={"txn_count_1h": 6, "amount": 499.99},
            feature_attributions=[fa],
            reason_codes=[rc],
            rule_matches=[rm],
            audit_trail=[audit],
        )

        assert resp.transaction.id == "c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f"
        assert resp.evaluation is not None
        assert resp.evaluation.risk_score == 85
        assert len(resp.features) == 2
        assert len(resp.feature_attributions) == 1
        assert resp.feature_attributions[0].shap_value == 1.452
        assert len(resp.reason_codes) == 1
        assert len(resp.rule_matches) == 1
        assert len(resp.audit_trail) == 1

    def test_transaction_detail_optional_evaluation_none(self) -> None:
        """Verify TransactionDetailResponse handles transactions without evaluations."""
        now = datetime(2026, 9, 17, 14, 30, 0, tzinfo=timezone.utc)
        meta = TransactionMetadataDetail(
            id="c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f",
            account_id="ACC_102938",
            merchant_category="retail",
            job_category="clergy",
            amount=50.0,
            cardholder_lat=30.0,
            cardholder_long=-90.0,
            merchant_lat=30.0,
            merchant_long=-90.0,
            city_pop=50000,
            transaction_timestamp=now,
            created_at=now,
        )

        resp = TransactionDetailResponse(
            transaction=meta,
            evaluation=None,
            features={"amount": 50.0},
            feature_attributions=[],
            reason_codes=[],
            rule_matches=[],
            audit_trail=[],
        )

        assert resp.evaluation is None
        assert resp.feature_attributions == []
        assert resp.reason_codes == []
        assert resp.rule_matches == []
        assert resp.audit_trail == []

    def test_feature_attribution_detail_boundaries(self) -> None:
        """Verify boundaries for FeatureAttributionDetail fields."""
        base_data = {
            "id": "f1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d",
            "feature_name": "amt_sum_24h",
            "display_name": "24 Hour Spending",
            "shap_value": 0.5,
            "direction": AttributionDirection.RISK_INCREASING,
            "relative_contribution_pct": 25.0,
            "rank": 1,
        }

        # Rank < 1
        with pytest.raises(ValidationError):
            FeatureAttributionDetail(**{**base_data, "rank": 0})

        # Contribution > 100
        with pytest.raises(ValidationError):
            FeatureAttributionDetail(**{**base_data, "relative_contribution_pct": 105.0})

        # Contribution < 0
        with pytest.raises(ValidationError):
            FeatureAttributionDetail(**{**base_data, "relative_contribution_pct": -1.0})

    def test_reason_code_detail_boundaries(self) -> None:
        """Verify boundaries for ReasonCodeDetail fields."""
        base_data = {
            "id": "r1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d",
            "code": "HIGH_AMOUNT",
            "headline": "High Amount",
            "description": "Amount is high.",
            "category": "AMOUNT",
            "source": ReasonSource.RULE,
            "severity": ReasonSeverity.HIGH,
            "rank": 1,
        }

        # Rank < 1
        with pytest.raises(ValidationError):
            ReasonCodeDetail(**{**base_data, "rank": 0})

    def test_rule_match_detail_boundaries(self) -> None:
        """Verify boundaries for RuleMatchDetail fields."""
        base_data = {
            "id": "m1a2b3c4-d5e6-7a8b-9c0d-1e2f3a4b5c6d",
            "rule_id": "RULE_TEST",
            "description": "Rule test",
            "feature_name": "amount",
            "operator": ">",
            "comparison_value": "1000",
            "outcome": RuleOutcome.BLOCK,
            "rule_type": RuleType.AMOUNT,
            "priority": 1,
        }

        # Priority < 1
        with pytest.raises(ValidationError):
            RuleMatchDetail(**{**base_data, "priority": 0})

    def test_transaction_detail_forbid_extra_fields(self) -> None:
        """Verify that extra unmodeled fields are rejected."""
        now = datetime.now(timezone.utc)
        meta_dict = {
            "id": "c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f",
            "account_id": "ACC_001",
            "merchant_category": "retail",
            "job_category": "tech",
            "amount": 25.0,
            "cardholder_lat": 0.0,
            "cardholder_long": 0.0,
            "merchant_lat": 0.0,
            "merchant_long": 0.0,
            "city_pop": 1000,
            "transaction_timestamp": now,
            "created_at": now,
            "card_pan_secret": "4111111111111111",  # Forbidden extra field
        }

        with pytest.raises(ValidationError):
            TransactionMetadataDetail(**meta_dict)

    def test_json_serialization_roundtrip(self) -> None:
        """Verify JSON export and re-import preserves all properties accurately."""
        now = datetime(2026, 9, 17, 14, 30, 0, tzinfo=timezone.utc)
        meta = TransactionMetadataDetail(
            id="c3d4e5f6-a7b8-4c5d-9e0f-1a2b3c4d5e6f",
            account_id="ACC_102938",
            merchant_category="retail",
            job_category="clergy",
            amount=50.0,
            cardholder_lat=30.0,
            cardholder_long=-90.0,
            merchant_lat=30.0,
            merchant_long=-90.0,
            city_pop=50000,
            transaction_timestamp=now,
            created_at=now,
        )

        resp = TransactionDetailResponse(
            transaction=meta,
            evaluation=None,
            features={"amount": 50.0, "is_fraud": 0},
            feature_attributions=[],
            reason_codes=[],
            rule_matches=[],
            audit_trail=[],
        )

        json_str = resp.model_dump_json()
        restored = TransactionDetailResponse.model_validate_json(json_str)
        assert restored.transaction.id == resp.transaction.id
        assert restored.features == resp.features


from backend.app.schemas.dashboard import (
    AnalyticsDistributionsResponse,
    AnalyticsRulesResponse,
    AnalyticsTrendsResponse,
    CategoryCount,
    DistributionBucket,
    RuleAnalyticsItem,
    RuleOutcomeBreakdown,
    TrendDataPoint,
    TrendInterval,
)


class TestAnalyticsSchemas:
    """Unit test suite for Increment 11.4 Analytics Schemas."""

    def test_valid_analytics_distributions_response(self) -> None:
        """Verify successful instantiation of AnalyticsDistributionsResponse."""
        rs_buckets = [
            DistributionBucket(bucket_label="0–9", lower_bound=0.0, upper_bound=9.0, count=50, percentage=50.0),
            DistributionBucket(bucket_label="10–19", lower_bound=10.0, upper_bound=19.0, count=50, percentage=50.0),
        ]
        ms_buckets = [
            DistributionBucket(bucket_label="0.0–0.1", lower_bound=0.0, upper_bound=0.1, count=100, percentage=100.0),
        ]
        tier_counts = [
            CategoryCount(category="LOW", count=90, percentage=90.0),
            CategoryCount(category="CRITICAL", count=10, percentage=10.0),
        ]
        decision_counts = [
            CategoryCount(category="APPROVE", count=90, percentage=90.0),
            CategoryCount(category="BLOCK", count=10, percentage=10.0),
        ]

        resp = AnalyticsDistributionsResponse(
            total_evaluated=100,
            risk_score_distribution=rs_buckets,
            model_score_distribution=ms_buckets,
            risk_tier_distribution=tier_counts,
            decision_distribution=decision_counts,
        )

        assert resp.total_evaluated == 100
        assert len(resp.risk_score_distribution) == 2
        assert resp.risk_score_distribution[0].bucket_label == "0–9"
        assert resp.risk_score_distribution[0].count == 50
        assert resp.risk_score_distribution[0].percentage == 50.0
        assert len(resp.risk_tier_distribution) == 2
        assert len(resp.decision_distribution) == 2

    def test_distribution_bucket_boundary_validations(self) -> None:
        """Verify boundary validations on DistributionBucket."""
        # Negative count -> ValidationError
        with pytest.raises(ValidationError):
            DistributionBucket(bucket_label="0–9", lower_bound=0.0, upper_bound=9.0, count=-1, percentage=0.0)

        # Percentage > 100.0 -> ValidationError
        with pytest.raises(ValidationError):
            DistributionBucket(bucket_label="0–9", lower_bound=0.0, upper_bound=9.0, count=10, percentage=101.0)

        # Percentage < 0.0 -> ValidationError
        with pytest.raises(ValidationError):
            DistributionBucket(bucket_label="0–9", lower_bound=0.0, upper_bound=9.0, count=10, percentage=-0.5)

        # Extra forbidden field -> ValidationError
        with pytest.raises(ValidationError):
            DistributionBucket(bucket_label="0–9", lower_bound=0.0, upper_bound=9.0, count=10, percentage=10.0, extra="forbid")

    def test_category_count_boundary_validations(self) -> None:
        """Verify boundary validations on CategoryCount."""
        # Negative count
        with pytest.raises(ValidationError):
            CategoryCount(category="BLOCK", count=-5, percentage=0.0)

        # Percentage > 100
        with pytest.raises(ValidationError):
            CategoryCount(category="BLOCK", count=5, percentage=110.0)

    def test_valid_analytics_trends_response(self) -> None:
        """Verify successful instantiation and structure of AnalyticsTrendsResponse."""
        t1 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 17, 13, 0, 0, tzinfo=timezone.utc)

        p1 = TrendDataPoint(
            timestamp=t1,
            total_count=50,
            total_amount=12500.00,
            average_risk_score=24.5,
            approval_count=45,
            review_count=3,
            block_count=2,
            high_critical_count=4,
        )
        p2 = TrendDataPoint(
            timestamp=t2,
            total_count=0,
            total_amount=0.0,
            average_risk_score=0.0,
            approval_count=0,
            review_count=0,
            block_count=0,
            high_critical_count=0,
        )

        resp = AnalyticsTrendsResponse(
            interval=TrendInterval.HOURLY,
            start_date=t1,
            end_date=t2,
            data_points=[p1, p2],
        )

        assert resp.interval == TrendInterval.HOURLY
        assert len(resp.data_points) == 2
        assert resp.data_points[0].total_count == 50
        assert resp.data_points[0].total_amount == 12500.00
        assert resp.data_points[1].total_count == 0

    def test_trend_data_point_boundary_validations(self) -> None:
        """Verify boundaries on TrendDataPoint."""
        t = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

        # Negative total_count
        with pytest.raises(ValidationError):
            TrendDataPoint(
                timestamp=t,
                total_count=-1,
                total_amount=0.0,
                average_risk_score=0.0,
                approval_count=0,
                review_count=0,
                block_count=0,
                high_critical_count=0,
            )

        # Score > 100
        with pytest.raises(ValidationError):
            TrendDataPoint(
                timestamp=t,
                total_count=10,
                total_amount=100.0,
                average_risk_score=105.0,
                approval_count=10,
                review_count=0,
                block_count=0,
                high_critical_count=0,
            )

    def test_valid_analytics_rules_response(self) -> None:
        """Verify successful instantiation of AnalyticsRulesResponse and RuleAnalyticsItem."""
        outcome = RuleOutcomeBreakdown(
            outcome=RuleOutcome.BLOCK,
            count=15,
            percentage=100.0,
        )
        rule_item = RuleAnalyticsItem(
            rule_id="RULE_VELOCITY_BURST",
            description="Hourly count exceeds 5",
            rule_type=RuleType.VELOCITY,
            priority=10,
            trigger_count=15,
            affected_transactions=15,
            trigger_rate=3.5,
            override_count=12,
            outcomes=[outcome],
        )

        resp = AnalyticsRulesResponse(
            total_rules_active=1,
            total_evaluations_analyzed=428,
            rules=[rule_item],
        )

        assert resp.total_rules_active == 1
        assert resp.total_evaluations_analyzed == 428
        assert len(resp.rules) == 1
        assert resp.rules[0].rule_id == "RULE_VELOCITY_BURST"
        assert resp.rules[0].trigger_count == 15
        assert resp.rules[0].override_count == 12
        assert len(resp.rules[0].outcomes) == 1

    def test_rule_analytics_item_boundaries(self) -> None:
        """Verify boundary validations on RuleAnalyticsItem."""
        # Priority < 1
        with pytest.raises(ValidationError):
            RuleAnalyticsItem(
                rule_id="RULE_TEST",
                description="Test rule",
                rule_type=RuleType.AMOUNT,
                priority=0,
                trigger_count=1,
                affected_transactions=1,
                trigger_rate=1.0,
                override_count=0,
                outcomes=[],
            )

        # Trigger rate > 100.0
        with pytest.raises(ValidationError):
            RuleAnalyticsItem(
                rule_id="RULE_TEST",
                description="Test rule",
                rule_type=RuleType.AMOUNT,
                priority=1,
                trigger_count=1,
                affected_transactions=1,
                trigger_rate=105.0,
                override_count=0,
                outcomes=[],
            )

    def test_analytics_json_serialization_roundtrip(self) -> None:
        """Verify serialization and deserialization roundtrip preserves exact values."""
        t1 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
        p = TrendDataPoint(
            timestamp=t1,
            total_count=10,
            total_amount=500.0,
            average_risk_score=45.0,
            approval_count=8,
            review_count=1,
            block_count=1,
            high_critical_count=2,
        )
        resp = AnalyticsTrendsResponse(
            interval=TrendInterval.DAILY,
            start_date=t1,
            end_date=t1,
            data_points=[p],
        )
        json_str = resp.model_dump_json()
        restored = AnalyticsTrendsResponse.model_validate_json(json_str)
        assert restored.interval == resp.interval
        assert len(restored.data_points) == 1
        assert restored.data_points[0].total_amount == 500.0


# ==============================================================================
# Increment 11.5: Simulation Schemas Unit Tests
# ==============================================================================

from backend.app.schemas.dashboard import (
    BaselineEvaluationSummary,
    FeatureDiffItem,
    RuleDiffItem,
    RuleDiffStatus,
    SimulatedEvaluationSummary,
    SimulationComparisonSummary,
    SimulationRequest,
    SimulationResponse,
)


class TestSimulationSchemas:
    """Unit test suite for Phase 11.5 Simulation Pydantic schemas."""

    def test_valid_simulation_request_without_baseline(self) -> None:
        """Verify valid SimulationRequest without a baseline transaction."""
        req = SimulationRequest(
            simulated_features={"amount": 500.0, "city_pop": 100000.0},
            top_k=5,
            top_mitigating=3,
            max_reasons=5,
        )
        assert req.baseline_transaction_id is None
        assert req.simulated_features == {"amount": 500.0, "city_pop": 100000.0}
        assert req.top_k == 5
        assert req.top_mitigating == 3
        assert req.max_reasons == 5

    def test_valid_simulation_request_with_baseline(self) -> None:
        """Verify valid SimulationRequest with a baseline transaction UUID."""
        req = SimulationRequest(
            baseline_transaction_id="550e8400-e29b-41d4-a716-446655440000",
            simulated_features={"amount": 1200.0},
            top_k=10,
            top_mitigating=5,
            max_reasons=8,
        )
        assert req.baseline_transaction_id == "550e8400-e29b-41d4-a716-446655440000"
        assert req.simulated_features["amount"] == 1200.0
        assert req.top_k == 10

    def test_simulation_request_extra_fields_forbidden(self) -> None:
        """Verify unexpected fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError) as exc:
            SimulationRequest(
                simulated_features={"amount": 100.0},
                unsupported_extra_field="malicious_payload",
            )
        assert "extra_forbidden" in str(exc.value)

    @pytest.mark.parametrize(
        "field,val",
        [
            ("top_k", 0),
            ("top_k", 51),
            ("top_mitigating", 0),
            ("top_mitigating", 51),
            ("max_reasons", 0),
            ("max_reasons", 51),
        ],
    )
    def test_simulation_request_bound_violations(self, field: str, val: int) -> None:
        """Verify boundary violations on top_k and max_reasons fail validation."""
        kwargs = {"simulated_features": {"amount": 100.0}, field: val}
        with pytest.raises(ValidationError):
            SimulationRequest(**kwargs)

    def test_feature_diff_item_valid(self) -> None:
        """Verify FeatureDiffItem model instantiates properly."""
        diff = FeatureDiffItem(
            feature_name="amount",
            display_name="Transaction Amount ($)",
            category="Spending & Monetary Volume",
            baseline_value=100.0,
            simulated_value=1500.0,
            is_modified=True,
            delta=1400.0,
        )
        assert diff.is_modified is True
        assert diff.delta == 1400.0

    def test_rule_diff_item_valid(self) -> None:
        """Verify RuleDiffItem model supports all RuleDiffStatus values."""
        for status in RuleDiffStatus:
            r = RuleDiffItem(
                rule_id="RULE_TEST_01",
                description="Test rule",
                rule_type="AMOUNT",
                priority=1,
                outcome="BLOCK",
                baseline_triggered=False,
                simulated_triggered=True,
                diff_status=status,
            )
            assert r.diff_status == status

    def test_simulation_response_serialization_roundtrip(self) -> None:
        """Verify full SimulationResponse serialization and deserialization roundtrip."""
        now = datetime.now(timezone.utc)
        resp = SimulationResponse(
            is_simulation=True,
            simulated_at=now,
            evaluation_latency_ms=12.5,
            baseline_transaction_id="550e8400-e29b-41d4-a716-446655440000",
            baseline=BaselineEvaluationSummary(
                transaction_id="550e8400-e29b-41d4-a716-446655440000",
                external_transaction_id="TX_BASELINE_01",
                risk_score=25.0,
                risk_tier="LOW",
                decision_action="APPROVE",
                model_score=0.15,
                is_overridden=False,
                rules_triggered_count=0,
            ),
            simulated=SimulatedEvaluationSummary(
                risk_score=85.0,
                risk_tier="HIGH",
                decision_action="BLOCK",
                model_score=0.82,
                base_value=-3.5,
                output_margin=1.52,
                is_overridden=False,
                decision_reason="Risk score 85.0 in high risk tier",
                rule_matches=[],
                reason_codes=[],
                feature_attributions=[],
            ),
            comparison=SimulationComparisonSummary(
                risk_score_delta=60.0,
                model_score_delta=0.67,
                tier_changed=True,
                action_changed=True,
                modified_features_count=3,
                feature_diffs=[],
                rule_diffs=[],
            ),
        )
        assert resp.is_simulation is True
        json_str = resp.model_dump_json()
        restored = SimulationResponse.model_validate_json(json_str)
        assert restored.simulated.risk_score == 85.0
        assert restored.comparison.risk_score_delta == 60.0
        assert restored.comparison.tier_changed is True


