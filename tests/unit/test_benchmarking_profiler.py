"""
Unit tests for PipelineProfiler and 7-stage micro-benchmarking infrastructure (Phase 10.1).
"""

from unittest.mock import AsyncMock, MagicMock
import pandas as pd
import pytest

from backend.app.benchmarking.metrics import PipelineStage
from backend.app.benchmarking.profiler import PipelineProfiler, ProfileResult
from backend.app.schemas.predict import TransactionPredictRequest
from ml.models.config import VAL_FEATURES_PATH


@pytest.fixture
def sample_transaction_dict() -> dict:
    """Load a real valid transaction dictionary from validation features Parquet."""
    df = pd.read_parquet(VAL_FEATURES_PATH)
    row_dict = df.iloc[0].to_dict()
    row_dict["transaction_id"] = "prof_tx_test_001"
    row_dict["account_id"] = "acc_user_123"
    row_dict["timestamp"] = "2026-09-15T12:00:00Z"
    return row_dict


class TestPipelineProfiler:
    """Tests for PipelineProfiler execution and stage timing accuracy."""

    def test_profiler_initialization(self) -> None:
        """Test default profiler initialization with RiskService."""
        profiler = PipelineProfiler()
        assert profiler.risk_service is not None
        assert profiler.evaluator is not None
        assert profiler.persistence_service is None

    def test_profile_transaction_sync(self, sample_transaction_dict: dict) -> None:
        """Test synchronous profiling of in-memory stages 1 through 6."""
        profiler = PipelineProfiler()
        result = profiler.profile_transaction(sample_transaction_dict)

        assert isinstance(result, ProfileResult)
        assert result.transaction_id == "prof_tx_test_001"
        assert result.total_profiled_latency_ms > 0.0
        assert 0.0 <= result.model_score <= 1.0
        assert 0 <= result.risk_score <= 100
        assert result.action in ("APPROVE", "REVIEW", "BLOCK")
        assert result.is_persisted is False

        # Verify all in-memory stages are present and non-negative
        timings = result.stage_latencies_ms
        assert PipelineStage.REQUEST_VALIDATION in timings
        assert PipelineStage.FEATURE_PREPARATION in timings
        assert PipelineStage.ML_INFERENCE in timings
        assert PipelineStage.TREESHAP_EXPLAINABILITY in timings
        assert PipelineStage.RULE_ENGINE_POLICY in timings
        assert PipelineStage.PERSISTENCE_MAPPING in timings

        for stage, lat in timings.items():
            assert isinstance(lat, float)
            assert lat >= 0.0, f"Stage {stage} latency must be non-negative, got {lat}"

    def test_profile_transaction_with_validated_request(self, sample_transaction_dict: dict) -> None:
        """Test profiling when passed a pre-validated TransactionPredictRequest."""
        req = TransactionPredictRequest.model_validate(sample_transaction_dict)
        profiler = PipelineProfiler()
        result = profiler.profile_transaction(req)

        assert result.transaction_id == "prof_tx_test_001"
        assert result.stage_latencies_ms[PipelineStage.REQUEST_VALIDATION] >= 0.0

    @pytest.mark.asyncio
    async def test_profile_transaction_async_without_persistence(
        self, sample_transaction_dict: dict
    ) -> None:
        """Test async profiling with skip_persistence=True."""
        profiler = PipelineProfiler()
        result = await profiler.profile_transaction_async(
            sample_transaction_dict, skip_persistence=True
        )

        assert result.is_persisted is False
        assert PipelineStage.POSTGRES_PERSISTENCE not in result.stage_latencies_ms

    @pytest.mark.asyncio
    async def test_profile_transaction_async_with_mocked_persistence(
        self, sample_transaction_dict: dict
    ) -> None:
        """Test async profiling with mocked persistence service."""
        mock_persistence = MagicMock()
        mock_persistence.persist_evaluation = AsyncMock(return_value=None)

        profiler = PipelineProfiler(persistence_service=mock_persistence)
        result = await profiler.profile_transaction_async(
            sample_transaction_dict, skip_persistence=False
        )

        assert result.is_persisted is True
        assert PipelineStage.POSTGRES_PERSISTENCE in result.stage_latencies_ms
        assert result.stage_latencies_ms[PipelineStage.POSTGRES_PERSISTENCE] >= 0.0
        assert mock_persistence.persist_evaluation.called

    def test_profile_result_to_dict(self, sample_transaction_dict: dict) -> None:
        """Test ProfileResult dictionary serialization."""
        profiler = PipelineProfiler()
        result = profiler.profile_transaction(sample_transaction_dict)
        d = result.to_dict()

        assert d["transaction_id"] == "prof_tx_test_001"
        assert "stage_latencies_ms" in d
        assert "request_validation" in d["stage_latencies_ms"]
        assert "ml_inference" in d["stage_latencies_ms"]
        assert "total_profiled_latency_ms" in d
        assert "model_score" in d
        assert "risk_score" in d
