# Persistence Integration Tests Guide

## Overview
The persistence integration test suite (`tests/integration/test_persistence_service_integration.py`) validates the complete, end-to-end relational persistence workflow against a real PostgreSQL database instance using SQLAlchemy 2.0 AsyncSession and `FraudPersistenceService`.

---

## 1. What the Integration Tests Verify

1. **Complete Aggregate Ingestion**:
   - Persists a complete fraud evaluation aggregate across all 6 core persistence entities (`Transaction`, `RiskEvaluation`, `EvaluationRuleMatch`, `EvaluationReasonCode`, `EvaluationFeatureAttribution`, `AuditLog`).
   - Verifies table rows, foreign keys, numeric precision, timestamps, and JSON snapshots.

2. **Cross-Session Commit Visibility**:
   - Opens a completely independent `AsyncSession` to read back persisted records, proving that data was genuinely committed to PostgreSQL and not merely cached in memory.

3. **External Transaction ID Duplicate Handling**:
   - Confirms that attempting to persist a second transaction with the same `external_transaction_id` raises `PersistenceConflictError` and prevents creation of any child or evaluation records.

4. **Atomic Rollback on Partial Failures**:
   - Proves that when an error occurs during child entity staging, the entire transaction is rolled back defensively, leaving 0 uncommitted or partial rows in the database.

5. **Foreign-Key Integrity & RESTRICT Deletions**:
   - Verifies PostgreSQL raises `IntegrityError` when attempting invalid foreign key references or attempting to delete a `Transaction` that is referenced by active `RiskEvaluation` records (`ON DELETE RESTRICT`).

6. **Database Check Constraints**:
   - Verifies database-level check constraints:
     - `chk_transactions_amount_positive` (`amount >= 0.00`)
     - `chk_risk_evaluations_model_score` (`model_score >= 0.0 AND model_score <= 1.0`)
     - `chk_risk_evaluations_risk_score` (`risk_score >= 0 AND risk_score <= 100`)

7. **Data Type & Schema Fidelity**:
   - Verifies round-trip accuracy for 55-feature JSON/JSONB snapshots, `Decimal` numeric columns, timezone-aware UTC timestamps, and all domain enums.

8. **Unit of Work Transaction Boundaries**:
   - Tests explicit commit/rollback controls and verifies that uncommitted work within an async context block is never visible to separate sessions.

---

## 2. Database Requirements & Strategy

* **Required Backend**: PostgreSQL 14+ with asynchronous driver `postgresql+asyncpg://`.
* **SQLite Policy**: SQLite is **not** used as a silent fallback because it does not enforce PostgreSQL-specific JSONB, decimal numeric scale, native UUID handling, or strict constraint semantics.
* **Graceful Degradation**: If PostgreSQL is unreachable, tests skip cleanly with an informative skip reason rather than failing the suite.

---

## 3. Environment Variables
 
| Variable Name | Default / Example | Purpose |
| :--- | :--- | :--- |
| `TEST_DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_test_db` | Dedicated connection string for integration test PostgreSQL database. |
 
> [!IMPORTANT]
> `DATABASE_URL` (the application database) is intentionally **never** used as a fallback for tests. This prevents tests from accidentally running destructive table cleanup against a development or production database.
 
> [!CAUTION]
> **DO NOT point `TEST_DATABASE_URL` to a production database!**
> The integration test fixtures execute automated table cleanup (`TRUNCATE ... CASCADE`) between test runs to guarantee test isolation.

---

## 4. How to Configure the Test Database

1. Ensure PostgreSQL is running locally on port 5432 (or your configured port):
   ```bash
   psql -U postgres -h localhost -c "CREATE DATABASE fraud_intelligence_test_db;"
   ```

2. Export the environment variable (optional if using default localhost credentials):
   ```bash
   # Windows PowerShell
   $env:TEST_DATABASE_URL="postgresql+asyncpg://postgres:your_password@localhost:5432/fraud_intelligence_test_db"

   # Linux / macOS
   export TEST_DATABASE_URL="postgresql+asyncpg://postgres:your_password@localhost:5432/fraud_intelligence_test_db"
   ```

---

## 5. Running the Tests

### Run Only Persistence Integration Tests
```bash
python -m pytest tests/integration/test_persistence_service_integration.py -v
```
or using the pytest marker:
```bash
python -m pytest -m integration -v
```

### Run Full Test Suite (Unit + ML + Integration)
```bash
python -m pytest tests/ -v
```
or quiet summary:
```bash
pytest -q
```

---

## 6. Behavior When PostgreSQL is Unavailable

When no PostgreSQL instance is reachable at the configured test database URL, all integration tests automatically produce a clean `SKIPPED` status with the following message:

```text
SKIPPED [1] tests/conftest.py:75: PostgreSQL test database not available. Set TEST_DATABASE_URL (e.g. postgresql+asyncpg://postgres:postgres@localhost:5432/fraud_intelligence_test_db) or ensure PostgreSQL is running to execute integration tests.
```

This prevents build pipelines and local developers without local database instances from experiencing test suite failures.
