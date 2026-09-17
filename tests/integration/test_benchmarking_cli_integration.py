"""
Integration tests for benchmark CLI execution and JSON/CSV export (Increment 10.4).
"""

import json
from pathlib import Path
import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.app.db.session import get_db_session
from backend.app.main import app
from ml.models.config import VAL_FEATURES_PATH
from scripts.benchmark import parse_args, run_benchmark_cli

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestBenchmarkCLIIntegration:
    """Integration test suite executing scripts/benchmark.py in ASGI mode."""

    async def test_run_benchmark_cli_asgi_mode_with_json_and_csv_export(
        self,
        db_session: AsyncSession,
        pg_engine: AsyncEngine,
        tmp_path: Path,
    ) -> None:
        """Verify full CLI execution in ASGI mode exporting JSON and CSV artifacts."""
        session_factory = async_sessionmaker(
            bind=pg_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

        async def override_get_db_session():
            async with session_factory() as session:
                try:
                    yield session
                except Exception:
                    await session.rollback()
                    raise

        app.dependency_overrides[get_db_session] = override_get_db_session

        json_out = tmp_path / "cli_results.json"
        csv_out = tmp_path / "cli_results.csv"

        cli_args = [
            "--dataset", str(VAL_FEATURES_PATH),
            "--requests", "3",
            "--concurrency", "1,2",
            "--warmup", "2",
            "--output", str(json_out),
            "--csv", str(csv_out),
            "--quiet",
        ]

        parsed = parse_args(cli_args)
        exit_code = await run_benchmark_cli(parsed)

        app.dependency_overrides.clear()

        # 1. Verify success exit code
        assert exit_code == 0

        # 2. Verify JSON export file
        assert json_out.exists()
        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "suite_results" in data
        assert "concurrency_sweep" in data["suite_results"]
        assert "ablation" in data["suite_results"]
        assert "bottleneck_analysis" in data["suite_results"]

        # 3. Verify CSV export file
        assert csv_out.exists()
        with open(csv_out, "r", encoding="utf-8") as f:
            csv_text = f.read()
        assert "=== MULTI-CONCURRENCY SCALABILITY MATRIX ===" in csv_text
        assert "=== SEVEN-STAGE COMPONENT LATENCY BREAKDOWN ===" in csv_text

    async def test_run_benchmark_cli_nonexistent_dataset_returns_error_code(
        self, tmp_path: Path
    ) -> None:
        """Verify CLI returns exit code 1 when dataset path does not exist."""
        bad_args = parse_args([
            "--dataset", "nonexistent_features_path.parquet",
            "--requests", "5",
            "--quiet",
        ])
        exit_code = await run_benchmark_cli(bad_args)
        assert exit_code == 1
