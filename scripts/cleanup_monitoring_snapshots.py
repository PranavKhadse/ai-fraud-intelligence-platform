#!/usr/bin/env python
"""
Monitoring Snapshot Retention Cleanup CLI Script.

Enforces bounded retention policy on `model_monitoring_snapshots`:
- Deletes HOURLY snapshots older than 30 days (configurable).
- Deletes DAILY snapshots older than 365 days (configurable).

Guarantees:
- Deletes strictly from `model_monitoring_snapshots`.
- Zero writes or deletions to `transactions`, `risk_evaluations`, `cases`, or `audit_logs`.

Usage:
    python scripts/cleanup_monitoring_snapshots.py
    python scripts/cleanup_monitoring_snapshots.py --hourly-days 30 --daily-days 365
    python scripts/cleanup_monitoring_snapshots.py --dry-run
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
from backend.app.repositories.monitoring_repository import MonitoringRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("monitoring_cleanup_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce retention policy and clean up expired monitoring snapshots."
    )
    parser.add_argument(
        "--hourly-days",
        type=int,
        default=30,
        help="Retention period in days for HOURLY snapshots. Default: 30 days.",
    )
    parser.add_argument(
        "--daily-days",
        type=int,
        default=365,
        help="Retention period in days for DAILY snapshots. Default: 365 days.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate cleanup and print count of expired records without deleting.",
    )
    return parser.parse_args()


async def run_snapshot_cleanup(args: argparse.Namespace) -> int:
    now = datetime.now(timezone.utc)
    hourly_cutoff = now - timedelta(days=args.hourly_days)
    daily_cutoff = now - timedelta(days=args.daily_days)

    logger.info(
        f"Evaluating monitoring snapshot retention: "
        f"hourly_cutoff={hourly_cutoff.isoformat()} ({args.hourly_days}d), "
        f"daily_cutoff={daily_cutoff.isoformat()} ({args.daily_days}d), "
        f"dry_run={args.dry_run}"
    )

    session_factory = get_session_factory()

    try:
        async with session_factory() as session:
            repo = MonitoringRepository(session)

            if args.dry_run:
                # Count matching expired snapshots without deleting
                expired_hourly, count_h = await repo.list_snapshots(
                    window_type="HOURLY", end_time=hourly_cutoff, limit=10000
                )
                expired_daily, count_d = await repo.list_snapshots(
                    window_type="DAILY", end_time=daily_cutoff, limit=10000
                )
                total_expired = count_h + count_d
                print("\n" + "=" * 60)
                print("MONITORING SNAPSHOT RETENTION DRY-RUN PREVIEW")
                print("=" * 60)
                print(f"Expired HOURLY Snapshots (> {args.hourly_days}d): {count_h}")
                print(f"Expired DAILY Snapshots  (> {args.daily_days}d): {count_d}")
                print(f"Total Snapshots to Delete:          {total_expired}")
                print("=" * 60 + "\n")
                return 0

            # Execute bounded deletion
            deleted_count = await repo.cleanup_expired_snapshots(
                hourly_retention_days=args.hourly_days,
                daily_retention_days=args.daily_days,
                now=now,
            )
            await session.commit()

        print("\n" + "=" * 60)
        print("MONITORING SNAPSHOT RETENTION CLEANUP COMPLETED")
        print("=" * 60)
        print(f"Snapshots Cleaned:       {deleted_count}")
        print(f"Hourly Cutoff Date:      {hourly_cutoff.isoformat()}")
        print(f"Daily Cutoff Date:       {daily_cutoff.isoformat()}")
        print("Guarantees:              Zero mutations to business transaction/case tables.")
        print("=" * 60 + "\n")
        return 0
    except Exception as exc:
        logger.error(f"Snapshot cleanup failed: {exc}", exc_info=True)
        return 1
    finally:
        await close_db_engine()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run_snapshot_cleanup(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
