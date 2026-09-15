# Phase 9 — Database & Persistence (PostgreSQL) Completion Report

## Executive Summary

Phase 9 establishes the production-grade relational persistence and data integrity foundation for the **AI-Powered Fraud Detection & Risk Intelligence Platform**.

All transactions evaluated by the machine learning inference engine and deterministic rule framework are atomically persisted across six normalized PostgreSQL tables with strict ACID guarantees, foreign-key integrity, explicit Unit of Work transaction boundaries, and full idempotency protection.

---

## 1. Architectural Overview & Component Structure

```
                                  ┌─────────────────────────────────┐
                                  │      Client / Payment Gateway   │
                                  └────────────────┬────────────────┘
                                                   │ POST /predict
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FASTAPI SERVICE LAYER                                      │
│                                                                                                  │
│  ┌─────────────────────────┐     ┌───────────────────────────┐     ┌──────────────────────────┐  │
│  │ Request Validation (422)│────►│ Pre-Inference Replay Path │────►│  Idempotent 200 Replay   │  │
│  │ (TransactionPredictReq) │     │ (Check external_tx_id)    │     │ (Zero redundant ML run)  │  │
│  └───────────┬─────────────┘     └─────────────┬─────────────┘     └──────────────────────────┘  │
│              │                                 │ (If new tx)                                     │
│              ▼                                 ▼                                                 │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              ML INFERENCE & EXPLAINABILITY                                 │  │
│  │  ┌───────────────────────┐    ┌────────────────────────┐    ┌───────────────────────────┐  │  │
│  │  │ Feature Snapshot (55) │───►│ XGBoost Model (v1.0.0) │───►│ TreeSHAP Explainer & Rule │  │  │
│  │  │ (Point-in-Time Vector)│    │ (Calibrated Score 0-1) │    │ Reason Codes & Waterfall  │  │  │
│  │  └───────────────────────┘    └────────────────────────┘    └─────────────┬─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────┼────────────────┘  │
│                                                                              │                   │
│  ┌───────────────────────────────────────────────────────────────────────────▼────────────────┐  │
│  │                         PERSISTENCE MAPPING & ORCHESTRATION                                │  │
│  │  ┌───────────────────────────────┐               ┌──────────────────────────────────────┐  │  │
│  │  │     RiskPersistenceMapper     │──────────────►│       FraudPersistenceService        │  │  │
│  │  │ (Map Request/Response to Cmd) │               │   (Atomic Aggregate Orchestration)   │  │  │
│  │  └───────────────────────────────┘               └──────────────────┬───────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┼──────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┼─────────────────────────┘
                                                                         │
                                                                         ▼
                                                     ┌──────────────────────────────────────┐
                                                     │     FraudPersistenceUnitOfWork       │
                                                     │  (Transaction Boundary: commit/undo) │
                                                     └──────────────────┬───────────────────┘
                                                                        │
                                   ┌────────────────────────────────────┴────────────────────────────────────┐
                                   ▼                                                                         ▼
                  ┌─────────────────────────────────┐                                       ┌─────────────────────────────────┐
                  │       PostgreSQL Database       │                                       │   Repositories (SQLAlchemy 2.0) │
                  │                                 │                                       │                                 │
                  │ • transactions                  │◄──────────────────────────────────────│ • TransactionRepository         │
                  │ • risk_evaluations              │                                       │ • RiskEvaluationRepository      │
                  │ • evaluation_rule_matches       │                                       │ • RuleMatchRepository           │
                  │ • evaluation_reason_codes       │                                       │ • ReasonCodeRepository          │
                  │ • evaluation_feature_attributions│                                      │ • FeatureAttributionRepository  │
                  │ • audit_logs                    │                                       │ • AuditLogRepository            │
                  └─────────────────────────────────┘                                       └─────────────────────────────────┘
```

---

## 2. Core Entities & Relational Schema

The database schema is declared in `backend/app/db/models/` using SQLAlchemy 2.0 declarative mappings:

### 2.1 Table Specifications

1. **`transactions`**:
   - **Primary Key**: `id` (`UUID`, internal immutable key).
   - **External ID**: `external_transaction_id` (`VARCHAR(128)`, indexed, partial unique index `uq_transactions_external_tx_id` WHERE `external_transaction_id IS NOT NULL`).
   - **Identifiers**: `account_id` (`VARCHAR(128)`, `NOT NULL`), `merchant_id` (`VARCHAR(128)`).
   - **Categoricals & Geometry**: `merchant_category` (`VARCHAR(64)`), `job_category` (`VARCHAR(64)`), `amount` (`NUMERIC(15, 2)`), `currency` (`VARCHAR(3)`), coordinates (`NUMERIC(9, 6)`), `city_pop` (`INTEGER`).
   - **Timestamps**: `transaction_timestamp` (`TIMESTAMPTZ`), `created_at` (`TIMESTAMPTZ`), `updated_at` (`TIMESTAMPTZ`).
   - **Vector Snapshot**: `features_snapshot` (`JSONB` storing the complete 55-feature point-in-time state).
   - **Check Constraint**: `chk_transactions_amount_positive` (`amount >= 0.00`).

2. **`risk_evaluations`**:
   - **Primary Key**: `id` (`UUID`).
   - **Foreign Key**: `transaction_id` (`UUID` $\to$ `transactions.id`, `ON DELETE RESTRICT`).
   - **Model Telemetry**: `model_version` (`VARCHAR(32)`), `policy_mode` (`ENUM('TRI_TIER', 'BINARY_AUTO')`), `model_score` (`NUMERIC(8, 6)`), `risk_score` (`SMALLINT`), `risk_tier` (`ENUM('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')`), `decision_action` (`ENUM('APPROVE', 'REVIEW', 'BLOCK')`), `baseline_action`, `is_overridden` (`BOOLEAN`), `rule_action`.
   - **SHAP Margins**: `output_margin` (`NUMERIC(10, 6)`), `base_value` (`NUMERIC(10, 6)`), `evaluation_latency_ms` (`NUMERIC(8, 2)`).
   - **Check Constraints**: `chk_risk_evaluations_model_score` ($0.0 \le \text{model\_score} \le 1.0$), `chk_risk_evaluations_risk_score` ($0 \le \text{risk\_score} \le 100$).

3. **`evaluation_rule_matches`**:
   - **Foreign Key**: `evaluation_id` (`UUID` $\to$ `risk_evaluations.id`, `ON DELETE CASCADE`).
   - **Fields**: `rule_id` (`VARCHAR(64)`), `description`, `feature_name`, `operator`, `comparison_value`, `outcome` (`ENUM('BLOCK', 'REVIEW', 'MONITOR')`), `rule_type`, `priority` (`INTEGER`).

4. **`evaluation_reason_codes`**:
   - **Foreign Key**: `evaluation_id` (`UUID` $\to$ `risk_evaluations.id`, `ON DELETE CASCADE`).
   - **Fields**: `code` (`VARCHAR(64)`), `headline`, `description`, `category`, `source` (`ENUM('MODEL', 'RULE')`), `severity` (`ENUM('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO')`), `rank` (`INTEGER`).

5. **`evaluation_feature_attributions`**:
   - **Foreign Key**: `evaluation_id` (`UUID` $\to$ `risk_evaluations.id`, `ON DELETE CASCADE`).
   - **Fields**: `feature_name` (`VARCHAR(64)`), `display_name`, `raw_value` (`JSONB`), `shap_value` (`NUMERIC(10, 6)`), `direction` (`ENUM('RISK_INCREASING', 'MITIGATING')`), `relative_contribution_pct` (`NUMERIC(6, 4)`), `rank` (`INTEGER`).

6. **`audit_logs`**:
   - **Fields**: `event_type` (`VARCHAR(64)`), `entity_type` (`ENUM('RISK_EVALUATION', 'TRANSACTION')`), `entity_id` (`UUID`), `action`, `actor_type` (`ENUM('SYSTEM', 'USER', 'ANALYST')`), `actor_id`, `correlation_id` (`VARCHAR(64)`), `client_ip` (`VARCHAR(45)`), `payload` (`JSONB`), `event_timestamp` (`TIMESTAMPTZ`).

---

## 3. Alembic Migration History

Alembic migrations under `backend/alembic/versions/` govern database versioning:

1. **`0001_initial_core_tables.py`**:
   - Creates the 6 core tables with primary keys, foreign keys (`RESTRICT` on Transaction, `CASCADE` on child explanations), check constraints, and performance indexes.
2. **`0002_add_unique_index_external_tx_id.py`**:
   - Creates the partial unique index `uq_transactions_external_tx_id` on `transactions(external_transaction_id) WHERE external_transaction_id IS NOT NULL`.
   - Ensures concurrency safety and prevents duplicate external transactions.

---

## 4. Repositories & Unit of Work

* **Repositories (`backend/app/repositories/`)**:
  - `TransactionRepository`, `RiskEvaluationRepository`, `RuleMatchRepository`, `ReasonCodeRepository`, `FeatureAttributionRepository`, `AuditLogRepository`.
  - Provide typed async CRUD, eager loading via `selectinload`, and query isolation.
* **Unit of Work (`FraudPersistenceUnitOfWork`)**:
  - Encapsulates transaction boundary lifecycle on injected `AsyncSession`.
  - Exposes `commit()`, `rollback()`, `flush()`, and async context manager (`async with uow:`) with defensive rollback on exceptions.
* **Persistence Service (`FraudPersistenceService`)**:
  - Staging order: `Transaction` $\to$ `flush()` $\to$ `RiskEvaluation` $\to$ `flush()` $\to$ child collections & `AuditLog` $\to$ single atomic `commit()`.
  - Selectively handles `IntegrityError`: converts `uq_transactions_external_tx_id` violations into `PersistenceConflictError`; translates unrelated database violations into `PersistenceError`.

---

## 5. API-to-Persistence Integration & Idempotency Protection

* **Deterministic Request Fingerprinting**:
  - SHA-256 fingerprint generated from normalized canonical payload attributes (account, merchant, amounts, coordinates, timestamp, and all 55 features).
* **Pre-Inference Fast-Path**:
  - If `transaction_id` is provided and already exists in the database:
    - **Identical Payload**: Reconstructs `PredictionResponse` from stored records with zero ML inference; returns `200 OK`.
    - **Mutated/Conflicting Payload**: Rejects immediately with `409 Conflict`.
* **Concurrency Race Recovery**:
  - If concurrent requests race past pre-inference checks, the loser worker encounters `IntegrityError` at commit, performs defensive rollback, re-queries the committed transaction, verifies payload equivalence, and replays `200 OK` or raises `409 Conflict`.
* **Metadata & Context Tracking**:
  - Extracts `X-Correlation-ID`, `X-Request-ID`, and verified client IP (with spoofing protection via `extract_client_ip`).

---

## 6. Timestamp & Normalization Semantics

1. **Client Timestamp (`request.timestamp`)**: External occurrence time optionally supplied by client; verified for equivalence within 1.0s tolerance.
2. **Ingestion Fallback Timestamp (`transactions.transaction_timestamp`)**: Fallback generated at persistence time when omitted; retries treat omitted timestamp as absent external data without false conflicts.
3. **Evaluation Timestamp (`risk_evaluations.evaluated_at`)**: Model evaluation timestamp preserved in database and reconstructed on replay.
4. **Audit and Ingestion Timestamp (`created_at`)**: System table row creation time.
5. **Length Boundaries**: `transaction_id`, `account_id`, `merchant_id` accept up to 128 characters; $\ge 129$ characters returns `422 Unprocessable Entity`.

---

## 7. Verification & Test Coverage Summary

| Test Suite | File / Scope | Count | Result |
| :--- | :--- | :--- | :--- |
| **Alembic Foundation** | `tests/unit/test_alembic_foundation.py` | 17 | 17 Passed |
| **ORM Models** | `tests/unit/test_orm_models.py` | 33 | 33 Passed |
| **Database Session & Health** | `tests/unit/test_db_*.py` | 24 | 24 Passed |
| **Repositories** | `tests/unit/test_*repository*.py` | 37 | 37 Passed |
| **Unit of Work & Service** | `tests/unit/test_unit_of_work.py`, `test_persistence_service.py` | 31 | 31 Passed |
| **Mapper & Endpoint Hardening** | `tests/unit/test_risk_persistence_mapper.py`, `test_predict_endpoint_hardening.py` | 60 | 60 Passed |
| **Idempotency & Boundaries** | `tests/unit/test_predict_idempotency.py` | 29 | 29 Passed |
| **Persistence Integration** | `tests/integration/test_persistence_service_integration.py` | 8 | 8 Passed (Live PostgreSQL) |
| **API Persistence Integration** | `tests/integration/test_predict_persistence_integration.py` | 16 | 16 Passed (Live PostgreSQL) |
| **API Prediction Baseline** | `tests/integration/test_api_predict.py` | 11 | 11 Passed |
| **Machine Learning Suite** | `tests/ml/` | 358 | 358 Passed |
| **Total Full Repository Suite** | `pytest tests/` | **624** | **624 Passed, 0 Failures** |

---

## 8. Conclusion

Phase 9 is **100% complete, verified, and production-ready**. All requirements for relational schema design, Alembic migrations, repository and Unit of Work patterns, API persistence, idempotency protection, and PostgreSQL integration testing have been fulfilled.
