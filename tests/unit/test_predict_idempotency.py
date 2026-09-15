"""
Unit Tests for Phase 9.6 Increment 3: Idempotency & Duplicate-Request Protection.

Validates:
1. Canonical request fingerprinting (deterministic SHA-256 hash).
2. Payload equivalence verification (core attributes, amounts, coordinates, 55 features).
3. PredictionResponse reconstruction from database entities.
4. Whitespace / empty transaction ID rejection (HTTP 422).
5. Pre-inference replay bypass (returns original evaluation with zero ML re-evaluation).
6. Pre-inference conflicting payload detection (HTTP 409 with zero ML re-evaluation).
7. Concurrency race recovery (replay on match, 409 on conflict).
8. Omitted transaction_id behavior (evaluates and persists normally).
"""

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException, Request, status

from backend.app.api.v1.endpoints.predict import predict_transaction
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
from sqlalchemy.exc import IntegrityError
from backend.app.repositories.exceptions import PersistenceConflictError, PersistenceError
from backend.app.schemas.predict import (
    FeatureAttributionResponse,
    PredictionResponse,
    ReasonCodeResponse,
    RuleMatchResponse,
    TransactionPredictRequest,
)
from backend.app.services.persistence_service import FraudPersistenceService
from backend.app.services.risk_persistence_mapper import (
    RiskPersistenceMapper,
    compute_request_fingerprint,
    is_payload_equivalent,
    reconstruct_prediction_response,
)
from backend.app.services.risk_service import RiskService
from tests.unit.test_predict_endpoint_hardening import make_mock_request


# ==============================================================================
# Sample Fixtures
# ==============================================================================

@pytest.fixture
def sample_valid_request() -> TransactionPredictRequest:
    """Construct a full canonical 55-feature TransactionPredictRequest."""
    data = {
        "transaction_id": "TX_IDEMPOTENCY_001",
        "account_id": "ACC_998877",
        "amount": 149.99,
        "currency": "USD",
        "merchant_category": "electronics",
        "job_category": "engineer",
        "cardholder_lat": 37.7749,
        "cardholder_long": -122.4194,
        "merchant_lat": 37.7750,
        "merchant_long": -122.4180,
        "city_pop": 850000,
        "timestamp": "2026-09-15T10:30:00Z",
        "transaction_hour": 10,
        "day_of_week": 1,
        "day_of_month": 15,
        "month": 9,
        "week_of_year": 38,
        "is_weekend": 0,
        "is_night": 0,
        "hour_sin": 0.5,
        "hour_cos": -0.866,
        "day_of_week_sin": 0.781,
        "day_of_week_cos": 0.623,
        "txn_count_1h": 1.0,
        "txn_count_6h": 3.0,
        "txn_count_24h": 5.0,
        "txn_count_7d": 15.0,
        "txn_count_30d": 45.0,
        "time_since_prev_txn_seconds": 3600.0,
        "is_first_account_txn": 0,
        "amt_sum_1h": 149.99,
        "amt_sum_24h": 520.0,
        "amt_sum_7d": 1800.0,
        "amt_sum_30d": 6500.0,
        "amt_mean_24h": 104.0,
        "amt_mean_7d": 120.0,
        "amt_max_24h": 250.0,
        "amt_median_30d": 95.0,
        "historical_amount_mean": 110.0,
        "historical_amount_std": 35.0,
        "historical_amount_median": 95.0,
        "amount_zscore": 1.14,
        "amount_ratio_to_historical_mean": 1.36,
        "account_txn_count_before": 45.0,
        "account_total_spend_before": 6500.0,
        "account_avg_amount_before": 110.0,
        "account_max_amount_before": 400.0,
        "account_unique_merchant_count_before": 20.0,
        "account_unique_category_count_before": 8.0,
        "account_merchant_txn_count_before": 2.0,
        "account_category_txn_count_before": 5.0,
        "account_merchant_spend_before": 300.0,
        "account_category_spend_before": 750.0,
        "merchant_txn_count_before": 500.0,
        "category_txn_count_before": 12000.0,
        "cardholder_merchant_distance_km": 0.15,
        "distance_from_prev_merchant_km": 2.5,
        "implied_travel_speed_kmh": 2.5,
        "is_impossible_travel_speed": 0,
    }
    return TransactionPredictRequest(**data)


@pytest.fixture
def mock_persisted_transaction(sample_valid_request: TransactionPredictRequest):
    """Construct a mock Transaction entity matching the sample request."""
    tx = SimpleNamespace(
        id="uuid-tx-001",
        external_transaction_id=sample_valid_request.transaction_id,
        account_id=sample_valid_request.account_id,
        amount=Decimal(str(round(sample_valid_request.amount, 2))),
        currency=sample_valid_request.currency,
        merchant_category=sample_valid_request.merchant_category,
        job_category=sample_valid_request.job_category,
        cardholder_lat=Decimal(str(round(sample_valid_request.cardholder_lat, 6))),
        cardholder_long=Decimal(str(round(sample_valid_request.cardholder_long, 6))),
        merchant_lat=Decimal(str(round(sample_valid_request.merchant_lat, 6))),
        merchant_long=Decimal(str(round(sample_valid_request.merchant_long, 6))),
        merchant_id=None,
        city_pop=int(sample_valid_request.city_pop),
        transaction_timestamp=datetime(2026, 9, 15, 10, 30, 0, tzinfo=timezone.utc),
        features_snapshot=RiskPersistenceMapper.extract_features_snapshot(sample_valid_request),
    )

    # Attach mock evaluation
    eval_record = SimpleNamespace(
        id="uuid-eval-001",
        transaction_id=tx.id,
        model_version="1.0.0",
        policy_mode=PolicyMode.TRI_TIER,
        model_score=Decimal("0.125000"),
        risk_score=13,
        risk_tier=RiskTier.LOW,
        decision_action=DecisionAction.APPROVE,
        is_overridden=False,
        rule_action=None,
        decision_reason="Low risk score evaluated by champion model.",
        evaluated_at=datetime(2026, 9, 15, 10, 30, 0, tzinfo=timezone.utc),
        rule_matches=[],
        reason_codes=[
            SimpleNamespace(
                code="RC_NORMAL_AMOUNT",
                headline="Transaction amount is consistent with history",
                description="Transaction amount of $149.99 aligns with normal profile.",
                category="AMOUNT",
                source=ReasonSource.MODEL,
                severity=ReasonSeverity.INFO,
                rank=1,
            )
        ],
        feature_attributions=[
            SimpleNamespace(
                feature_name="amount_zscore",
                display_name="Amount Z-Score",
                raw_value=1.14,
                shap_value=Decimal("0.024500"),
                direction=AttributionDirection.RISK_INCREASING,
                relative_contribution_pct=Decimal("0.6000"),
                rank=1,
            ),
            SimpleNamespace(
                feature_name="cardholder_merchant_distance_km",
                display_name="Distance to Merchant",
                raw_value=0.15,
                shap_value=Decimal("-0.052000"),
                direction=AttributionDirection.MITIGATING,
                relative_contribution_pct=Decimal("0.8500"),
                rank=1,
            ),
        ],
    )
    tx.evaluations = [eval_record]
    return tx


# ==============================================================================
# 1. Request Fingerprinting & Equivalence Tests
# ==============================================================================

class TestRequestFingerprintAndEquivalence:
    """Validates deterministic fingerprinting and payload equivalence checking."""

    def test_fingerprint_deterministic_for_identical_payloads(self, sample_valid_request):
        """Identical requests produce the exact same 64-character SHA-256 fingerprint."""
        h1 = compute_request_fingerprint(sample_valid_request)
        h2 = compute_request_fingerprint(sample_valid_request)
        assert len(h1) == 64
        assert h1 == h2

    def test_fingerprint_changes_on_amount_difference(self, sample_valid_request):
        """Fingerprint changes when amount is modified."""
        h1 = compute_request_fingerprint(sample_valid_request)
        modified = sample_valid_request.model_copy(update={"amount": 999.00})
        h2 = compute_request_fingerprint(modified)
        assert h1 != h2

    def test_fingerprint_changes_on_account_id_difference(self, sample_valid_request):
        """Fingerprint changes when account_id is modified."""
        h1 = compute_request_fingerprint(sample_valid_request)
        modified = sample_valid_request.model_copy(update={"account_id": "ACC_DIFFERENT_999"})
        h2 = compute_request_fingerprint(modified)
        assert h1 != h2

    def test_fingerprint_changes_on_feature_vector_difference(self, sample_valid_request):
        """Fingerprint changes when any of the 55 predictive features is modified."""
        h1 = compute_request_fingerprint(sample_valid_request)
        modified = sample_valid_request.model_copy(update={"txn_count_1h": 12.0})
        h2 = compute_request_fingerprint(modified)
        assert h1 != h2

    def test_is_payload_equivalent_returns_true_for_matching_data(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """is_payload_equivalent returns True when all transaction attributes match."""
        assert is_payload_equivalent(sample_valid_request, mock_persisted_transaction) is True

    def test_is_payload_equivalent_returns_false_on_account_mismatch(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """is_payload_equivalent returns False when account_id differs."""
        modified = sample_valid_request.model_copy(update={"account_id": "ACC_CHANGED"})
        assert is_payload_equivalent(modified, mock_persisted_transaction) is False

    def test_is_payload_equivalent_returns_false_on_amount_mismatch(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """is_payload_equivalent returns False when amount differs."""
        modified = sample_valid_request.model_copy(update={"amount": 500.00})
        assert is_payload_equivalent(modified, mock_persisted_transaction) is False

    def test_is_payload_equivalent_returns_false_on_feature_mismatch(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """is_payload_equivalent returns False when a feature in the snapshot differs."""
        modified = sample_valid_request.model_copy(update={"amount_zscore": 9.5})
        assert is_payload_equivalent(modified, mock_persisted_transaction) is False


# ==============================================================================
# 2. Response Reconstruction Tests
# ==============================================================================

class TestResponseReconstruction:
    """Validates reconstructing canonical PredictionResponse from persisted entities."""

    def test_reconstructs_full_prediction_response(self, mock_persisted_transaction):
        """Reconstructs complete PredictionResponse with matching fields, reason codes, and factors."""
        resp = reconstruct_prediction_response(mock_persisted_transaction)
        assert isinstance(resp, PredictionResponse)
        assert resp.transaction_id == "TX_IDEMPOTENCY_001"
        assert resp.decision_action == "APPROVE"
        assert resp.risk_tier == "LOW"
        assert resp.risk_score == 13
        assert resp.model_score == 0.125
        assert resp.model_version == "1.0.0"
        assert resp.evaluated_at == "2026-09-15T10:30:00+00:00"
        assert len(resp.reason_codes) == 1
        assert resp.reason_codes[0].code == "RC_NORMAL_AMOUNT"
        assert len(resp.top_risk_factors) == 1
        assert resp.top_risk_factors[0].feature_name == "amount_zscore"
        assert len(resp.top_mitigating_factors) == 1
        assert resp.top_mitigating_factors[0].feature_name == "cardholder_merchant_distance_km"

    def test_reconstruct_raises_value_error_if_no_evaluations(self):
        """Raises ValueError when transaction has no evaluations attached."""
        empty_tx = SimpleNamespace(id="uuid-no-eval", external_transaction_id="TX_EMPTY", evaluations=[])
        with pytest.raises(ValueError, match="no associated evaluations"):
            reconstruct_prediction_response(empty_tx)


# ==============================================================================
# 3. Endpoint Idempotency Handling Tests
# ==============================================================================

class TestEndpointIdempotencyExecution:
    """Validates /predict endpoint pre-inference replay, conflict detection, and race recovery."""

    @pytest.mark.asyncio
    async def test_empty_or_whitespace_transaction_id_returns_422(self, sample_valid_request):
        """Whitespace-only transaction_id is rejected with HTTP 422."""
        sample_valid_request.transaction_id = "   "
        mock_risk = MagicMock(spec=RiskService)
        mock_pers = MagicMock(spec=FraudPersistenceService)
        http_req = make_mock_request()

        with pytest.raises(HTTPException) as exc_info:
            await predict_transaction(
                request=sample_valid_request,
                http_request=http_req,
                risk_service=mock_risk,
                persistence_service=mock_pers,
            )
        assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "cannot be empty or whitespace-only" in exc_info.value.detail
        mock_risk.predict_transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_idempotent_replay_bypasses_ml_inference(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """When identical transaction exists, returns original response without calling RiskService."""
        mock_risk = MagicMock(spec=RiskService)
        mock_pers = MagicMock(spec=FraudPersistenceService)
        mock_pers.get_existing_evaluation = AsyncMock(return_value=mock_persisted_transaction)
        mock_pers.persist_evaluation = AsyncMock()

        http_req = make_mock_request()
        response = await predict_transaction(
            request=sample_valid_request,
            http_request=http_req,
            risk_service=mock_risk,
            persistence_service=mock_pers,
        )

        assert response.transaction_id == sample_valid_request.transaction_id
        assert response.decision_action == "APPROVE"
        assert response.risk_score == 13
        # Must NOT have called RiskService (bypassed CPU-bound ML inference)
        mock_risk.predict_transaction.assert_not_called()
        # Must NOT have called persist_evaluation (zero redundant database writes)
        mock_pers.persist_evaluation.assert_not_called()

    @pytest.mark.asyncio
    async def test_conflicting_payload_reuse_returns_409_before_inference(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """When existing transaction exists with different amount, returns 409 without calling RiskService."""
        # Mutate request amount to create conflict
        sample_valid_request.amount = 999.99

        mock_risk = MagicMock(spec=RiskService)
        mock_pers = MagicMock(spec=FraudPersistenceService)
        mock_pers.get_existing_evaluation = AsyncMock(return_value=mock_persisted_transaction)

        http_req = make_mock_request()
        with pytest.raises(HTTPException) as exc_info:
            await predict_transaction(
                request=sample_valid_request,
                http_request=http_req,
                risk_service=mock_risk,
                persistence_service=mock_pers,
            )

        assert exc_info.value.status_code == status.HTTP_409_CONFLICT
        assert "conflicting request payload" in exc_info.value.detail
        mock_risk.predict_transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_race_resolves_to_idempotent_replay(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """When persist_evaluation encounters race PersistenceConflictError, re-queries DB and replays."""
        mock_risk = MagicMock(spec=RiskService)
        mock_response = reconstruct_prediction_response(mock_persisted_transaction)
        mock_risk.predict_transaction.return_value = mock_response

        mock_pers = MagicMock(spec=FraudPersistenceService)
        # 1st call (pre-check) returns None, 2nd call (post-conflict recovery) returns committed entity
        mock_pers.get_existing_evaluation = AsyncMock(side_effect=[None, mock_persisted_transaction])
        mock_pers.persist_evaluation = AsyncMock(side_effect=PersistenceConflictError("Duplicate key"))

        http_req = make_mock_request()
        response = await predict_transaction(
            request=sample_valid_request,
            http_request=http_req,
            risk_service=mock_risk,
            persistence_service=mock_pers,
        )

        assert response.transaction_id == sample_valid_request.transaction_id
        assert response.decision_action == "APPROVE"

    @pytest.mark.asyncio
    async def test_transaction_id_exceeding_128_chars_returns_422(self, sample_valid_request):
        """Transaction identifier exceeding 128 characters is rejected with HTTP 422."""
        sample_valid_request.transaction_id = "T" * 129
        mock_risk = MagicMock(spec=RiskService)
        mock_pers = MagicMock(spec=FraudPersistenceService)
        http_req = make_mock_request()

        with pytest.raises(HTTPException) as exc_info:
            await predict_transaction(
                request=sample_valid_request,
                http_request=http_req,
                risk_service=mock_risk,
                persistence_service=mock_pers,
            )
        assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "exceeds maximum length of 128 characters" in exc_info.value.detail
        mock_risk.predict_transaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_omitted_transaction_id_processes_normally(self, sample_valid_request):
        """When transaction_id is None, it bypasses pre-inference lookup, evaluates ML, and persists."""
        sample_valid_request.transaction_id = None
        mock_risk = MagicMock(spec=RiskService)
        mock_response = PredictionResponse(
            transaction_id=None,
            model_score=0.1,
            risk_score=10,
            risk_tier="LOW",
            decision_action="APPROVE",
            policy_mode="TRI_TIER",
            reason="Low risk",
            is_overridden=False,
            model_version="1.0.0",
            evaluated_at="2026-09-15T10:30:00+00:00",
        )
        mock_risk.predict_transaction.return_value = mock_response
        mock_pers = MagicMock(spec=FraudPersistenceService)
        mock_pers.persist_evaluation = AsyncMock()

        http_req = make_mock_request()
        resp = await predict_transaction(
            request=sample_valid_request,
            http_request=http_req,
            risk_service=mock_risk,
            persistence_service=mock_pers,
        )

        assert resp.transaction_id is None
        assert resp.decision_action == "APPROVE"
        mock_pers.get_existing_evaluation.assert_not_called()
        mock_risk.predict_transaction.assert_called_once()
        mock_pers.persist_evaluation.assert_called_once()


# ==============================================================================
# 4. Fingerprint Completeness & Ordering Tests
# ==============================================================================

class TestFingerprintCompletenessAndOrdering:
    """Validates that all decision-relevant fields affect fingerprinting and equivalence."""

    def test_currency_difference_changes_fingerprint_and_fails_equivalence(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """Currency modification changes fingerprint and fails equivalence check."""
        h1 = compute_request_fingerprint(sample_valid_request)
        req_eur = sample_valid_request.model_copy(update={"currency": "EUR"})
        h2 = compute_request_fingerprint(req_eur)
        assert h1 != h2
        assert is_payload_equivalent(req_eur, mock_persisted_transaction) is False

    def test_merchant_id_difference_changes_fingerprint_and_fails_equivalence(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """Merchant ID modification changes fingerprint and fails equivalence check."""
        req1 = sample_valid_request.model_copy(update={"merchant_id": "MERCHANT_AAA"})
        req2 = sample_valid_request.model_copy(update={"merchant_id": "MERCHANT_BBB"})
        assert compute_request_fingerprint(req1) != compute_request_fingerprint(req2)

        mock_persisted_transaction.merchant_id = "MERCHANT_AAA"
        assert is_payload_equivalent(req1, mock_persisted_transaction) is True
        assert is_payload_equivalent(req2, mock_persisted_transaction) is False

    def test_coordinates_difference_changes_fingerprint_and_fails_equivalence(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """Geographical coordinate differences change fingerprint and fail equivalence check."""
        req_lat = sample_valid_request.model_copy(update={"cardholder_lat": 38.0000})
        assert compute_request_fingerprint(sample_valid_request) != compute_request_fingerprint(req_lat)
        assert is_payload_equivalent(req_lat, mock_persisted_transaction) is False

    def test_timestamp_difference_fails_equivalence(
        self, sample_valid_request, mock_persisted_transaction
    ):
        """Explicitly different timestamp fails equivalence check."""
        mock_persisted_transaction.transaction_timestamp = datetime(2026, 9, 15, 10, 30, 0, tzinfo=timezone.utc)
        req_diff_time = sample_valid_request.model_copy(update={"timestamp": "2026-09-14T10:30:00Z"})
        assert is_payload_equivalent(req_diff_time, mock_persisted_transaction) is False

    def test_rule_match_priority_ordering_preserved_on_reconstruction(
        self, mock_persisted_transaction
    ):
        """Rule matches are reconstructed strictly ordered by evaluation priority."""
        eval_rec = mock_persisted_transaction.evaluations[0]
        rm_low = SimpleNamespace(
            rule_id="RULE_VELOCITY_WARN",
            description="Velocity elevated",
            feature_name="txn_count_1h",
            operator=">",
            comparison_value="5",
            outcome=RuleOutcome.REVIEW,
            rule_type=RuleType.VELOCITY,
            priority=20,
        )
        rm_high = SimpleNamespace(
            rule_id="RULE_GEO_BLOCK",
            description="Impossible travel speed",
            feature_name="implied_travel_speed_kmh",
            operator=">",
            comparison_value="800",
            outcome=RuleOutcome.BLOCK,
            rule_type=RuleType.GEOGRAPHY,
            priority=5,
        )
        # Add out-of-order in DB record
        eval_rec.rule_matches = [rm_low, rm_high]

        reconstructed = reconstruct_prediction_response(mock_persisted_transaction)
        assert reconstructed.rules_triggered == ["RULE_GEO_BLOCK", "RULE_VELOCITY_WARN"]
        assert reconstructed.rule_matches[0].rule_id == "RULE_GEO_BLOCK"
        assert reconstructed.rule_matches[0].priority == 5
        assert reconstructed.rule_matches[1].rule_id == "RULE_VELOCITY_WARN"
        assert reconstructed.rule_matches[1].priority == 20


class TestIdNormalizationAndLengthBoundaries:
    """Validates boundary length and whitespace normalization rules for domain identifiers."""

    def test_exact_128_char_ids_accepted(self, sample_valid_request):
        """Exact 128-character transaction_id, account_id, and merchant_id are accepted."""
        id_128 = "A" * 128
        req = sample_valid_request.model_copy(
            update={
                "transaction_id": id_128,
                "account_id": id_128,
                "merchant_id": id_128,
            }
        )
        assert req.transaction_id == id_128
        assert req.account_id == id_128
        assert req.merchant_id == id_128
        assert len(req.transaction_id) == 128

    def test_129_char_transaction_id_rejected_by_schema(self, sample_valid_request):
        """129-character transaction_id fails Pydantic schema validation."""
        data = sample_valid_request.model_dump()
        data["transaction_id"] = "T" * 129
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            TransactionPredictRequest(**data)
        assert "transaction_id" in str(exc_info.value)

    def test_129_char_account_id_rejected_by_schema(self, sample_valid_request):
        """129-character account_id fails Pydantic schema validation."""
        data = sample_valid_request.model_dump()
        data["account_id"] = "A" * 129
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            TransactionPredictRequest(**data)
        assert "account_id" in str(exc_info.value)

    def test_129_char_merchant_id_rejected_by_schema(self, sample_valid_request):
        """129-character merchant_id fails Pydantic schema validation."""
        data = sample_valid_request.model_dump()
        data["merchant_id"] = "M" * 129
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            TransactionPredictRequest(**data)
        assert "merchant_id" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_whitespace_only_transaction_id_raises_422(self, sample_valid_request):
        """Whitespace-only transaction_id raises HTTP 422 in predict endpoint."""
        req = sample_valid_request.model_copy(update={"transaction_id": "   "})
        mock_risk_service = MagicMock(spec=RiskService)
        mock_persistence_service = AsyncMock(spec=FraudPersistenceService)
        mock_http = make_mock_request()

        with pytest.raises(HTTPException) as exc_info:
            await predict_transaction(
                request=req,
                http_request=mock_http,
                risk_service=mock_risk_service,
                persistence_service=mock_persistence_service,
            )
        assert exc_info.value.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "cannot be empty or whitespace-only" in exc_info.value.detail

    def test_leading_trailing_whitespace_normalization_in_fingerprint(self, sample_valid_request):
        """Leading and trailing whitespace in transaction_id does not create divergent fingerprints."""
        req_clean = sample_valid_request.model_copy(update={"account_id": "ACC_12345"})
        req_padded = sample_valid_request.model_copy(update={"account_id": "  ACC_12345  "})
        assert compute_request_fingerprint(req_clean) == compute_request_fingerprint(req_padded)


class TestIntegrityErrorDiscriminator:
    """Validates selective translation of IntegrityError based on violation constraint."""

    @pytest.mark.asyncio
    async def test_duplicate_key_integrity_error_translates_to_conflict(self, sample_valid_request):
        """IntegrityError from unique index violation is translated to PersistenceConflictError."""
        from backend.app.services.risk_persistence_mapper import RiskPersistenceMapper, RiskEvaluationContext

        mock_uow = AsyncMock()
        mock_uow.transactions.exists_by_external_id.return_value = False
        mock_uow.commit.side_effect = IntegrityError(
            "INSERT INTO transactions ...",
            {},
            Exception("duplicate key value violates unique constraint \"uq_transactions_external_tx_id\""),
        )

        service = FraudPersistenceService(mock_uow)
        cmd = RiskPersistenceMapper.map_prediction_to_command(
            request=sample_valid_request,
            response=PredictionResponse(
                transaction_id="TX_IDEMPOTENCY_001",
                model_score=0.12,
                risk_score=12,
                risk_tier="LOW",
                decision_action="APPROVE",
                policy_mode="TRI_TIER",
                reason="Low risk",
                is_overridden=False,
                rule_action=None,
                rules_triggered=[],
                rule_matches=[],
                reason_codes=[],
                top_risk_factors=[],
                top_mitigating_factors=[],
                model_version="1.0.0",
                evaluated_at="2026-09-15T10:30:00Z",
            ),
            context=RiskEvaluationContext(external_transaction_id="TX_IDEMPOTENCY_001", account_id="ACC_998877"),
        )

        with pytest.raises(PersistenceConflictError) as exc_info:
            await service.persist_evaluation(cmd)

        assert "already exists" in str(exc_info.value)
        mock_uow.rollback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unrelated_check_constraint_integrity_error_translates_to_persistence_error(
        self, sample_valid_request
    ):
        """IntegrityError from unrelated check constraint failure translates to PersistenceError (not conflict)."""
        from backend.app.services.risk_persistence_mapper import RiskPersistenceMapper, RiskEvaluationContext

        mock_uow = AsyncMock()
        mock_uow.transactions.exists_by_external_id.return_value = False
        mock_uow.commit.side_effect = IntegrityError(
            "INSERT INTO transactions ...",
            {},
            Exception("new row violates check constraint \"chk_transactions_amount_positive\""),
        )

        service = FraudPersistenceService(mock_uow)
        cmd = RiskPersistenceMapper.map_prediction_to_command(
            request=sample_valid_request,
            response=PredictionResponse(
                transaction_id="TX_IDEMPOTENCY_001",
                model_score=0.12,
                risk_score=12,
                risk_tier="LOW",
                decision_action="APPROVE",
                policy_mode="TRI_TIER",
                reason="Low risk",
                is_overridden=False,
                rule_action=None,
                rules_triggered=[],
                rule_matches=[],
                reason_codes=[],
                top_risk_factors=[],
                top_mitigating_factors=[],
                model_version="1.0.0",
                evaluated_at="2026-09-15T10:30:00Z",
            ),
            context=RiskEvaluationContext(external_transaction_id="TX_IDEMPOTENCY_001", account_id="ACC_998877"),
        )

        with pytest.raises(PersistenceError) as exc_info:
            await service.persist_evaluation(cmd)

        # Must NOT be PersistenceConflictError
        assert not isinstance(exc_info.value, PersistenceConflictError)
        assert "Database integrity constraint violation" in str(exc_info.value)
        mock_uow.rollback.assert_awaited_once()

