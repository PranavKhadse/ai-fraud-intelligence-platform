#!/usr/bin/env python
"""
Audit Log Cryptographic Integrity Verification CLI.

Scans the append-only `audit_logs` table in deterministic chronological order,
validating SHA-256 hash chaining, detection of broken links, altered payloads,
missing/reordered records, and unchained injections.

Usage:
    python scripts/verify_audit_integrity.py
    python scripts/verify_audit_integrity.py --limit 1000
    python scripts/verify_audit_integrity.py --json
    python scripts/verify_audit_integrity.py --verbose
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.db.session import close_db_engine, get_session_factory
from backend.app.services.audit_verifier import AuditIntegrityVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_verifier_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify cryptographic hash chain integrity across audit logs."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of audit records to verify (chronological scan).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON result.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Output detailed breakdown of anomalies and verification scope.",
    )
    return parser.parse_args()


async def run_verification(args: argparse.Namespace) -> int:
    session_factory = get_session_factory()
    verifier = AuditIntegrityVerifier()

    try:
        async with session_factory() as session:
            result = await verifier.verify_database(session, limit=args.limit)

        if args.json:
            output = {
                "is_valid": result.is_valid,
                "total_records": result.total_records,
                "verified_records": result.verified_records,
                "legacy_records": result.legacy_records,
                "first_chained_id": str(result.first_chained_id) if result.first_chained_id else None,
                "last_verified_id": str(result.last_verified_id) if result.last_verified_id else None,
                "last_verified_hash": result.last_verified_hash,
                "verified_at": result.verified_at.isoformat(),
                "errors": [
                    {
                        "error_type": err.error_type.value,
                        "message": err.message,
                        "record_id": str(err.record_id) if err.record_id else None,
                        "record_index": err.record_index,
                        "expected": err.expected,
                        "actual": err.actual,
                    }
                    for err in result.errors
                ],
                "limitations": result.limitations,
            }
            print(json.dumps(output, indent=2))
        else:
            print("\n" + "=" * 70)
            print(" AUDIT LOG CRYPTOGRAPHIC INTEGRITY VERIFICATION REPORT")
            print("=" * 70)
            print(f" Status            : {'[PASS] VALID' if result.is_valid else '[FAIL] TAMPERING DETECTED'}")
            print(f" Total Records     : {result.total_records}")
            print(f" Verified Chained  : {result.verified_records}")
            print(f" Legacy Unchained  : {result.legacy_records}")
            print(f" Anomalies Found   : {len(result.errors)}")
            if result.last_verified_hash:
                print(f" Latest Head Hash  : {result.last_verified_hash}")
            print("-" * 70)

            if not result.is_valid:
                print("\n[!] INTEGRITY ERRORS:")
                for i, err in enumerate(result.errors, 1):
                    print(f"  {i}. [{err.error_type.value}] Index {err.record_index} (Record {err.record_id})")
                    print(f"     Details: {err.message}")
                    if err.expected or err.actual:
                        print(f"     Expected: {err.expected}")
                        print(f"     Actual:   {err.actual}")
                print()

            if args.verbose:
                print("\nDocumented Verification Limitations:")
                for lim in result.limitations:
                    print(f"  - {lim}")
                print()

            print("=" * 70)

        return 0 if result.is_valid else 1

    finally:
        await close_db_engine()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run_verification(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
