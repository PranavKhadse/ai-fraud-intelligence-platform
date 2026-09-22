#!/usr/bin/env python
"""
Monitoring Snapshot Generation CLI Script.

Executes windowed monitoring calculations and idempotently persists a snapshot
into the `model_monitoring_snapshots` table.

Usage:
    python scripts/generate_monitoring_snapshot.py --window-type DAILY
    python scripts/generate_monitoring_snapshot.py --window-type HOURLY
    python scripts/generate_monitoring_snapshot.py --window-type DAILY --start-time 2026-09-20T00:00:00Z --end-time 2026-09-21T00:00:00Z
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.session import close_db_engine, get_session_factory
from backend.app.services.monitoring_service import MonitoringService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitoring_snapshot_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and persist model monitoring snapshot rollups."
    )
    parser.add_argument(
        "--window-type",
        type=str,
        choices=["HOURLY", "DAILY", "hourly", "daily"],
        default="DAILY",
        help="Rollup window type ('HOURLY' or 'DAILY'). Default: DAILY.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Optional ISO UTC window start timestamp (e.g. '2026-09-21T00:00:00Z').",
    )
    parser.add_argument(
        "--end-time",
        type=str,
        default=None,
        help="Optional ISO UTC window end timestamp (e.g. '2026-09-22T00:00:00Z').",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="1.0.0",
        help="Target model release version. Default: 1.0.0.",
    )
    return parser.parse_args()


async def run_snapshot_generation(args: argparse.Namespace) -> int:
    win_type = args.window_type.upper().strip()
    now = datetime.now(timezone.utc)

    if args.start_time and args.end_time:
        start_dt = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(args.end_time.replace("Z", "+00:00"))
    elif win_type == "HOURLY":
        # Default to previous full hour
        end_dt = now.replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(hours=1)
    else:  # DAILY
        # Default to previous full day
        end_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=1)

    logger.info(
        f"Initiating snapshot generation: window_type={win_type} range=[{start_dt.isoformat()} -> {end_dt.isoformat()}] "
        f"model_version={args.model_version}"
    )

    service = MonitoringService()
    session_factory = get_session_factory()

    try:
        async with session_factory() as session:
            snapshot = await service.compute_and_persist_snapshot(
                session=session,
                window_type=win_type,
                window_start=start_dt,
                window_end=end_dt,
                model_version=args.model_version,
            )

        print("\n" + "=" * 60)
        print("MONITORING SNAPSHOT GENERATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Snapshot ID:             {snapshot.id}")
        print(f"Model Version:           {snapshot.model_version}")
        print(f"Window Type:             {snapshot.window_type}")
        print(f"Window Start:            {snapshot.window_start.isoformat()}")
        print(f"Window End:              {snapshot.window_end.isoformat()}")
        print(f"Evaluated Transactions:  {snapshot.sample_count}")
        print(f"Resolved Labeled Cases:  {snapshot.labeled_count}")
        print(f"Overall Health Status:   {snapshot.overall_status}")
        print(f"  - Data Drift Status:   {snapshot.data_drift_status}")
        print(f"  - Prediction Drift:    {snapshot.prediction_drift_status}")
        print(f"  - Model Performance:   {snapshot.performance_status}")
        print("=" * 60 + "\n")
        return 0
    except Exception as exc:
        logger.error(f"Snapshot generation failed: {exc}", exc_info=True)
        return 1
    finally:
        await close_db_engine()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run_snapshot_generation(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
