"""
Automated Unit Tests for Phase 6 Increment 1 & 3: Risk Engine Normalization & Hardened Decision Policy.

Validates:
1. DecisionPolicyConfig: Defaults, validation of bounds, type safety, boolean rejection, ordering, immutability,
   to_dict serialization, from_dict parsing, unknown keys rejection, string enum parsing.
2. normalize_model_score: Scalar normalization, exact boundary rounding, strict range checking, error handling.
3. normalize_model_scores: Vectorized normalization, shape/order preservation, dtype validation, error handling.
4. risk_tier_from_score: Exact tier boundaries (0, 34, 35, 59, 60, 77, 78, 100), type checking, error handling.
5. DecisionPolicyEngine: Policy routing under TRI_TIER and BINARY_AUTO modes, threshold inclusivity, zero-width review band,
   batch evaluation, decision provenance (thresholds_applied, model_version).
6. DecisionResult: Immutability, field integrity, provenance fields, JSON serialization via to_dict().
"""

import math
import pytest
import numpy as np

from ml.risk_engine.config import (
    DecisionAction,
    RiskTier,
    PolicyMode,
    DecisionPolicyConfig,
)
from ml.risk_engine.normalization import (
    normalize_model_score,
    normalize_model_scores,
    risk_tier_from_score,
)
from ml.risk_engine.policy import (
    DecisionResult,
    DecisionPolicyEngine,
)


# =====================================================================
# 1. DecisionPolicyConfig Tests
# =====================================================================

class TestDecisionPolicyConfig:
    """Test suite for DecisionPolicyConfig creation, validation, immutability, and serialization."""

    def test_default_configuration(self) -> None:
        """Verify default configuration parameters reflect Phase 5 cost-optimal baseline."""
        config = DecisionPolicyConfig()
        assert config.policy_mode == PolicyMode.TRI_TIER
        assert config.review_threshold == 0.35
        assert config.block_threshold == 0.78

    def test_custom_configuration(self) -> None:
        """Verify custom configuration is properly stored."""
        config = DecisionPolicyConfig(
            policy_mode=PolicyMode.BINARY_AUTO,
            review_threshold=0.20,
            block_threshold=0.60,
        )
        assert config.policy_mode == PolicyMode.BINARY_AUTO
        assert config.review_threshold == 0.20
        assert config.block_threshold == 0.60

    def test_configuration_immutability(self) -> None:
        """Verify DecisionPolicyConfig is a frozen dataclass."""
        config = DecisionPolicyConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.block_threshold = 0.90  # type: ignore

    def test_invalid_policy_mode_type(self) -> None:
        """Verify non-PolicyMode enum raises TypeError."""
        with pytest.raises(TypeError, match="policy_mode must be an instance of PolicyMode"):
            DecisionPolicyConfig(policy_mode="TRI_TIER")  # type: ignore

    @pytest.mark.parametrize("invalid_threshold", [-0.01, -1.0, 1.01, 2.5])
    def test_out_of_bounds_review_threshold(self, invalid_threshold: float) -> None:
        """Verify review_threshold outside [0.0, 1.0] raises ValueError."""
        with pytest.raises(ValueError, match="review_threshold must be in"):
            DecisionPolicyConfig(review_threshold=invalid_threshold, block_threshold=0.9)

    @pytest.mark.parametrize("invalid_threshold", [-0.01, -1.0, 1.01, 2.5])
    def test_out_of_bounds_block_threshold(self, invalid_threshold: float) -> None:
        """Verify block_threshold outside [0.0, 1.0] raises ValueError."""
        with pytest.raises(ValueError, match="block_threshold must be in"):
            DecisionPolicyConfig(review_threshold=0.1, block_threshold=invalid_threshold)

    def test_threshold_ordering_violation(self) -> None:
        """Verify review_threshold > block_threshold raises ValueError."""
        with pytest.raises(ValueError, match="cannot exceed block_threshold"):
            DecisionPolicyConfig(review_threshold=0.85, block_threshold=0.75)

    def test_equal_thresholds_allowed(self) -> None:
        """Verify review_threshold == block_threshold is allowed (zero-width review band)."""
        config = DecisionPolicyConfig(review_threshold=0.78, block_threshold=0.78)
        assert config.review_threshold == 0.78
        assert config.block_threshold == 0.78

    @pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_thresholds(self, non_finite: float) -> None:
        """Verify NaN or infinite thresholds raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            DecisionPolicyConfig(review_threshold=non_finite, block_threshold=0.8)
        with pytest.raises(ValueError, match="must be finite"):
            DecisionPolicyConfig(review_threshold=0.3, block_threshold=non_finite)

    @pytest.mark.parametrize("bool_val", [True, False])
    def test_boolean_threshold_rejection(self, bool_val: bool) -> None:
        """Verify boolean values are explicitly rejected."""
        with pytest.raises(TypeError, match="cannot be a boolean value"):
            DecisionPolicyConfig(review_threshold=bool_val, block_threshold=0.8)  # type: ignore
        with pytest.raises(TypeError, match="cannot be a boolean value"):
            DecisionPolicyConfig(review_threshold=0.3, block_threshold=bool_val)  # type: ignore

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict produces clean JSON-serializable dictionary."""
        config = DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.35,
            block_threshold=0.78,
        )
        d = config.to_dict()
        assert d == {
            "policy_mode": "TRI_TIER",
            "review_threshold": 0.35,
            "block_threshold": 0.78,
        }

    def test_from_dict_roundtrip(self) -> None:
        """Verify from_dict reconstitutes identical configuration."""
        original = DecisionPolicyConfig(
            policy_mode=PolicyMode.BINARY_AUTO,
            review_threshold=0.15,
            block_threshold=0.85,
        )
        reconstituted = DecisionPolicyConfig.from_dict(original.to_dict())
        assert reconstituted == original

    def test_from_dict_string_enum_parsing(self) -> None:
        """Verify from_dict parses string enum representations."""
        config1 = DecisionPolicyConfig.from_dict({"policy_mode": "BINARY_AUTO"})
        assert config1.policy_mode == PolicyMode.BINARY_AUTO

        config2 = DecisionPolicyConfig.from_dict({"policy_mode": "TRI_TIER"})
        assert config2.policy_mode == PolicyMode.TRI_TIER

    def test_from_dict_unknown_keys_rejection(self) -> None:
        """Verify from_dict rejects unexpected dictionary keys."""
        with pytest.raises(ValueError, match="Unknown configuration key"):
            DecisionPolicyConfig.from_dict({"invalid_key": 123})

    def test_from_dict_non_dict_rejection(self) -> None:
        """Verify from_dict rejects non-dict inputs."""
        with pytest.raises(TypeError, match="data must be a dict"):
            DecisionPolicyConfig.from_dict(["policy_mode", "TRI_TIER"])  # type: ignore

    def test_from_dict_invalid_string_enum_rejection(self) -> None:
        """Verify from_dict rejects invalid string mode values."""
        with pytest.raises(ValueError, match="Invalid policy_mode"):
            DecisionPolicyConfig.from_dict({"policy_mode": "INVALID_MODE"})


# =====================================================================
# 2. Scalar Normalization Tests
# =====================================================================

class TestScalarNormalization:
    """Test suite for normalize_model_score."""

    @pytest.mark.parametrize(
        "model_score, expected_risk_score",
        [
            (0.0, 0),
            (0.004, 0),
            (0.006, 1),
            (0.35, 35),
            (0.50, 50),
            (0.779, 78),
            (0.78, 78),
            (0.94, 94),
            (0.996, 100),
            (1.0, 100),
        ],
    )
    def test_benchmark_conversions(self, model_score: float, expected_risk_score: int) -> None:
        """Verify precise conversion of canonical float model scores to integer risk scores."""
        result = normalize_model_score(model_score)
        assert isinstance(result, int)
        assert not isinstance(result, bool)
        assert result == expected_risk_score

    def test_return_type_and_bounds(self) -> None:
        """Verify output is strictly an int in [0, 100]."""
        for s in np.linspace(0.0, 1.0, 101):
            score = normalize_model_score(s)
            assert isinstance(score, int)
            assert 0 <= score <= 100

    @pytest.mark.parametrize("invalid_val", [-0.001, -0.5, -10.0, 1.001, 1.5, 100.0])
    def test_out_of_bounds_rejection(self, invalid_val: float) -> None:
        """Verify model scores outside [0.0, 1.0] raise ValueError."""
        with pytest.raises(ValueError, match="must be within"):
            normalize_model_score(invalid_val)

    @pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejection(self, non_finite: float) -> None:
        """Verify NaN / +/-inf raise ValueError."""
        with pytest.raises(ValueError, match="must be a finite number"):
            normalize_model_score(non_finite)

    @pytest.mark.parametrize("bool_val", [True, False])
    def test_boolean_rejection(self, bool_val: bool) -> None:
        """Verify boolean inputs raise TypeError."""
        with pytest.raises(TypeError, match="cannot be a boolean value"):
            normalize_model_score(bool_val)  # type: ignore

    @pytest.mark.parametrize("invalid_type", ["0.5", None, [0.5], {"score": 0.5}])
    def test_non_numeric_rejection(self, invalid_type: object) -> None:
        """Verify non-numeric types raise TypeError."""
        with pytest.raises(TypeError, match="must be numeric"):
            normalize_model_score(invalid_type)  # type: ignore


# =====================================================================
# 3. Vectorized Normalization Tests
# =====================================================================

class TestVectorizedNormalization:
    """Test suite for normalize_model_scores."""

    def test_vector_conversion_and_shape(self) -> None:
        """Verify 1D array conversion matches expected values and dtype."""
        inputs = np.array([0.0, 0.35, 0.50, 0.78, 0.94, 1.0], dtype=np.float64)
        expected = np.array([0, 35, 50, 78, 94, 100], dtype=np.int64)
        output = normalize_model_scores(inputs)
        assert isinstance(output, np.ndarray)
        assert output.shape == inputs.shape
        assert output.dtype == np.int64
        np.testing.assert_array_equal(output, expected)

    def test_monotonicity(self) -> None:
        """Verify monotonic non-decreasing output for sorted inputs."""
        inputs = np.linspace(0.0, 1.0, 500)
        output = normalize_model_scores(inputs)
        assert np.all(np.diff(output) >= 0)

    def test_empty_array_rejection(self) -> None:
        """Verify empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_model_scores(np.array([]))

    def test_multidimensional_array_rejection(self) -> None:
        """Verify 2D or 3D arrays raise ValueError."""
        with pytest.raises(ValueError, match="must be a 1-dimensional array"):
            normalize_model_scores(np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_boolean_array_rejection(self) -> None:
        """Verify boolean array raises TypeError."""
        with pytest.raises(TypeError, match="cannot be a boolean array"):
            normalize_model_scores(np.array([True, False, True]))

    def test_non_array_rejection(self) -> None:
        """Verify lists or scalars raise TypeError."""
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            normalize_model_scores([0.1, 0.5, 0.9])  # type: ignore

    def test_nan_and_inf_in_array_rejection(self) -> None:
        """Verify NaN or inf inside array raises ValueError."""
        with pytest.raises(ValueError, match="contains NaN or infinite"):
            normalize_model_scores(np.array([0.1, np.nan, 0.8]))
        with pytest.raises(ValueError, match="contains NaN or infinite"):
            normalize_model_scores(np.array([0.1, np.inf, 0.8]))

    def test_out_of_range_in_array_rejection(self) -> None:
        """Verify array with out-of-range elements raises ValueError."""
        with pytest.raises(ValueError, match="must be within"):
            normalize_model_scores(np.array([-0.05, 0.5, 0.8]))
        with pytest.raises(ValueError, match="must be within"):
            normalize_model_scores(np.array([0.1, 0.5, 1.05]))


# =====================================================================
# 4. Semantic Risk Tier Tests
# =====================================================================

class TestRiskTierMapping:
    """Test suite for risk_tier_from_score."""

    @pytest.mark.parametrize(
        "score, expected_tier",
        [
            (0, RiskTier.LOW),
            (15, RiskTier.LOW),
            (34, RiskTier.LOW),
            (35, RiskTier.MEDIUM),
            (45, RiskTier.MEDIUM),
            (59, RiskTier.MEDIUM),
            (60, RiskTier.HIGH),
            (70, RiskTier.HIGH),
            (77, RiskTier.HIGH),
            (78, RiskTier.CRITICAL),
            (90, RiskTier.CRITICAL),
            (100, RiskTier.CRITICAL),
        ],
    )
    def test_boundary_tier_mappings(self, score: int, expected_tier: RiskTier) -> None:
        """Verify exact boundary mapping to semantic risk tiers."""
        assert risk_tier_from_score(score) == expected_tier

    @pytest.mark.parametrize("bool_val", [True, False])
    def test_boolean_rejection(self, bool_val: bool) -> None:
        """Verify booleans raise TypeError."""
        with pytest.raises(TypeError, match="cannot be a boolean value"):
            risk_tier_from_score(bool_val)  # type: ignore

    @pytest.mark.parametrize("non_int", [35.0, 78.5, "50", None])
    def test_non_integer_rejection(self, non_int: object) -> None:
        """Verify non-integers (including floats) raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            risk_tier_from_score(non_int)  # type: ignore

    @pytest.mark.parametrize("out_of_bounds", [-1, -50, 101, 200])
    def test_out_of_bounds_rejection(self, out_of_bounds: int) -> None:
        """Verify integers outside [0, 100] raise ValueError."""
        with pytest.raises(ValueError, match="must be an integer between 0 and 100"):
            risk_tier_from_score(out_of_bounds)


# =====================================================================
# 5. Decision Policy Engine & Routing Tests
# =====================================================================

class TestDecisionPolicyEngine:
    """Test suite for DecisionPolicyEngine routing, modes, and batch evaluation."""

    def test_engine_initialization_defaults(self) -> None:
        """Verify default engine setup uses default DecisionPolicyConfig."""
        engine = DecisionPolicyEngine()
        assert engine.config.policy_mode == PolicyMode.TRI_TIER
        assert engine.config.review_threshold == 0.35
        assert engine.config.block_threshold == 0.78
        assert engine.model_version is None

    def test_invalid_config_type(self) -> None:
        """Verify passing an invalid config type raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of DecisionPolicyConfig"):
            DecisionPolicyEngine(config="default")  # type: ignore

    @pytest.mark.parametrize(
        "model_score, expected_action, expected_risk_score, expected_tier",
        [
            (0.00, DecisionAction.APPROVE, 0, RiskTier.LOW),
            (0.10, DecisionAction.APPROVE, 10, RiskTier.LOW),
            (0.34, DecisionAction.APPROVE, 34, RiskTier.LOW),
            (0.35, DecisionAction.REVIEW, 35, RiskTier.MEDIUM),
            (0.50, DecisionAction.REVIEW, 50, RiskTier.MEDIUM),
            (0.77, DecisionAction.REVIEW, 77, RiskTier.HIGH),
            (0.78, DecisionAction.BLOCK, 78, RiskTier.CRITICAL),
            (0.94, DecisionAction.BLOCK, 94, RiskTier.CRITICAL),
            (1.00, DecisionAction.BLOCK, 100, RiskTier.CRITICAL),
        ],
    )
    def test_tri_tier_policy_routing(
        self,
        model_score: float,
        expected_action: DecisionAction,
        expected_risk_score: int,
        expected_tier: RiskTier,
    ) -> None:
        """Verify complete tri-tier decision routing at all critical threshold points."""
        engine = DecisionPolicyEngine(model_version="1.0.0")
        res = engine.evaluate(model_score)
        assert res.action == expected_action
        assert res.risk_score == expected_risk_score
        assert res.risk_tier == expected_tier
        assert res.model_score == model_score
        assert res.policy_mode == PolicyMode.TRI_TIER
        assert res.model_version == "1.0.0"
        assert res.thresholds_applied == {"review_threshold": 0.35, "block_threshold": 0.78}
        assert isinstance(res.reason, str) and len(res.reason) > 0

    @pytest.mark.parametrize(
        "model_score, expected_action",
        [
            (0.00, DecisionAction.APPROVE),
            (0.35, DecisionAction.APPROVE),
            (0.50, DecisionAction.APPROVE),
            (0.77, DecisionAction.APPROVE),
            (0.78, DecisionAction.BLOCK),
            (0.94, DecisionAction.BLOCK),
            (1.00, DecisionAction.BLOCK),
        ],
    )
    def test_binary_auto_policy_routing(self, model_score: float, expected_action: DecisionAction) -> None:
        """Verify binary auto decision routing routes all scores below block_threshold to APPROVE."""
        config = DecisionPolicyConfig(policy_mode=PolicyMode.BINARY_AUTO, block_threshold=0.78)
        engine = DecisionPolicyEngine(config=config)
        res = engine.evaluate(model_score)
        assert res.action == expected_action
        assert res.policy_mode == PolicyMode.BINARY_AUTO

    def test_zero_width_review_band(self) -> None:
        """Verify routing when review_threshold == block_threshold (zero-width review band)."""
        config = DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.78,
            block_threshold=0.78,
        )
        engine = DecisionPolicyEngine(config=config)
        res_below = engine.evaluate(0.779)
        assert res_below.action == DecisionAction.APPROVE
        assert "zero-width review band" in res_below.reason

        res_at = engine.evaluate(0.78)
        assert res_at.action == DecisionAction.BLOCK

    def test_custom_thresholds_routing(self) -> None:
        """Verify routing with custom thresholds."""
        config = DecisionPolicyConfig(
            policy_mode=PolicyMode.TRI_TIER,
            review_threshold=0.20,
            block_threshold=0.60,
        )
        engine = DecisionPolicyEngine(config=config)
        assert engine.evaluate(0.19).action == DecisionAction.APPROVE
        assert engine.evaluate(0.20).action == DecisionAction.REVIEW
        assert engine.evaluate(0.59).action == DecisionAction.REVIEW
        assert engine.evaluate(0.60).action == DecisionAction.BLOCK

    def test_batch_evaluation(self) -> None:
        """Verify batch evaluation preserves ordering and outputs."""
        engine = DecisionPolicyEngine(model_version="1.0.0")
        scores = np.array([0.10, 0.40, 0.85], dtype=np.float64)
        results = engine.evaluate_batch(scores)
        assert len(results) == 3
        assert results[0].action == DecisionAction.APPROVE
        assert results[1].action == DecisionAction.REVIEW
        assert results[2].action == DecisionAction.BLOCK
        for r in results:
            assert r.model_version == "1.0.0"
            assert r.thresholds_applied == {"review_threshold": 0.35, "block_threshold": 0.78}


# =====================================================================
# 6. DecisionResult Structure & Serialization Tests
# =====================================================================

class TestDecisionResult:
    """Test suite for DecisionResult dataclass deep immutability, mapping compatibility, and serialization."""

    def test_result_immutability(self) -> None:
        """Verify DecisionResult fields and nested mapping cannot be mutated."""
        engine = DecisionPolicyEngine()
        res = engine.evaluate(0.50)

        # 1. Top-level attribute mutation is rejected
        with pytest.raises(Exception):
            res.action = DecisionAction.BLOCK  # type: ignore

        # 2. Nested mapping in-place item assignment is rejected
        with pytest.raises((TypeError, Exception)):
            res.thresholds_applied["block_threshold"] = 0.99  # type: ignore

        # 3. Nested mapping key deletion is rejected
        with pytest.raises((TypeError, Exception)):
            del res.thresholds_applied["review_threshold"]  # type: ignore

    def test_thresholds_applied_mapping_compatibility(self) -> None:
        """Verify read-only MappingProxyType supports standard dictionary read operations."""
        engine = DecisionPolicyEngine()
        res = engine.evaluate(0.50)

        # Subscription read
        assert res.thresholds_applied["review_threshold"] == 0.35
        assert res.thresholds_applied["block_threshold"] == 0.78

        # In operator
        assert "review_threshold" in res.thresholds_applied
        assert "non_existent" not in res.thresholds_applied

        # .get() method
        assert res.thresholds_applied.get("review_threshold") == 0.35
        assert res.thresholds_applied.get("non_existent", 99.0) == 99.0

        # Iteration & length
        assert len(res.thresholds_applied) == 2
        assert set(res.thresholds_applied.keys()) == {"review_threshold", "block_threshold"}

    def test_to_dict_deep_independence(self) -> None:
        """Verify to_dict produces a normal dictionary that does not mutate the source object."""
        engine = DecisionPolicyEngine(model_version="1.0.0")
        res = engine.evaluate(0.78)
        d = res.to_dict()

        # Mutate the dictionary returned by to_dict
        d["thresholds_applied"]["block_threshold"] = 0.99
        d["action"] = "APPROVE"

        # Ensure original DecisionResult is unaffected
        assert res.thresholds_applied["block_threshold"] == 0.78
        assert res.action == DecisionAction.BLOCK

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict produces a clean dictionary with string enum values and provenance."""
        engine = DecisionPolicyEngine(model_version="1.0.0")
        res = engine.evaluate(0.78)
        d = res.to_dict()
        assert d == {
            "action": "BLOCK",
            "risk_score": 78,
            "risk_tier": "CRITICAL",
            "model_score": 0.78,
            "policy_mode": "TRI_TIER",
            "reason": (
                "Model score (0.7800) meets or exceeds block threshold (0.7800). "
                "Action: BLOCK."
            ),
            "thresholds_applied": {
                "review_threshold": 0.35,
                "block_threshold": 0.78,
            },
            "model_version": "1.0.0",
        }
