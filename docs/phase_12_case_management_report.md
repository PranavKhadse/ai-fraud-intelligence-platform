# Phase 12 Milestone Report: Human Review & Case Management Platform

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
> **Milestone:** Phase 12 — Human Review & Case Management System  
> **Version:** 1.0.0-phase12  
> **Status:** 🟢 **COMPLETED & VERIFIED**  
> **Author:** Antigravity AI & Senior Fraud Intelligence Architecture Team  
> **Date:** September 2026  

---

## 1. Executive Summary

Phase 12 delivers an enterprise-grade, concurrency-safe **Human Review & Case Management System** for the AI-Powered Fraud Detection & Risk Intelligence Platform. It bridges real-time automated ML risk scoring and human investigator decision-making by establishing:

1. **Relational Persistence Foundation**: Normalized PostgreSQL tables (`cases`, `case_notes`) with strict foreign keys, enums, table-level `UNIQUE(transaction_id)` constraint enforcing strict 1-to-1 cardinality, and Alembic migration `0003_add_cases_and_case_notes_tables.py`.
2. **Automated & Manual Case Escalation**: Automatic `OPEN` case staging on `REVIEW` policy decisions inside `FraudPersistenceService`, alongside manual analyst/admin escalation workflows with pre-check validation and collision-safe `CASE-YYYYMMDD-XXXXXX` numbering.
3. **Robust Lifecycle State Machine**: Formal states (`OPEN`, `IN_REVIEW`, `ESCALATED`, `RESOLVED`, `CLOSED`) with strict row-locking mutation boundaries (`SELECT FOR UPDATE`), immutable audit logging (`AuditEntityType.CASE`), and structured human disposition recording.
4. **Comprehensive REST API**: 10 production REST endpoints under `/api/v1/cases` providing review queue querying, queue KPI metrics, full investigation detail, notes management, lifecycle transitions, and authoritative case audit timelines.
5. **Interactive React Investigation Workspace**: High-performance React 18 + TypeScript workspace featuring a multi-dimensional Review Queue, 7 real-time KPI metrics, point-in-time 55-feature snapshot inspectability, explainable reason codes, SHAP waterfall visualizations, and action modals.
6. **Ground-Truth Data Integrity**: Guaranteed persistence of point-in-time feature snapshots, calibrated risk scores, model scores, and human review outcomes for downstream Phase 14 model retraining loops.

---

## 2. Architecture & Data Model

```
+-------------------------------------------------------------------------------------------------------+
|                                    TRANSACTION & RISK EVALUATION                                     |
|  +-----------------------------+                           +---------------------------------------+  |
|  |        Transaction          |                           |            RiskEvaluation             |  |
|  |-----------------------------|                           |---------------------------------------|  |
|  | id: UUID (PK)               |<--------------------+     | id: UUID (PK)                         |  |
|  | external_transaction_id: str|                     |     | transaction_id: UUID (FK)             |  |
|  | amount: Decimal             |                     |     | model_score: Decimal                  |  |
|  | features_snapshot: JSONB    |                     |     | risk_score: int                       |  |
|  +-----------------------------+                     |     | decision_action: DecisionAction (ENUM)|  |
|                                                      |     +---------------------------------------+  |
+------------------------------------------------------|------------------------------------------------+
                                                       |                                 |
                                  1-to-1 (UNIQUE)      |                                 | 1-to-1
                                                       v                                 v
+-------------------------------------------------------------------------------------------------------+
|                                         CASE MANAGEMENT DOMAIN                                        |
|  +-------------------------------------------------------------------------------------------------+  |
|  |                                              Case                                               |  |
|  |-------------------------------------------------------------------------------------------------|  |
|  | id: UUID (PK)                                                                                   |  |
|  | case_number: VARCHAR(32) (UNIQUE)                                                               |  |
|  | transaction_id: UUID (FK -> transactions.id, UNIQUE)                                            |  |
|  | evaluation_id: UUID (FK -> risk_evaluations.id)                                                 |  |
|  | status: CaseStatus (OPEN, IN_REVIEW, ESCALATED, RESOLVED, CLOSED)                                 |  |
|  | priority: CasePriority (LOW, MEDIUM, HIGH, CRITICAL)                                            |  |
|  | trigger_source: CaseTriggerSource (AUTOMATED_REVIEW_POLICY, MANUAL_ANALYST_ESCALATION)          |  |
|  | assigned_to: VARCHAR(128) (NULLable)                                                            |  |
|  | assigned_at: TIMESTAMPTZ (NULLable)                                                             |  |
|  | opened_at: TIMESTAMPTZ                                                                          |  |
|  | resolved_at: TIMESTAMPTZ (NULLable)                                                             |  |
|  | closed_at: TIMESTAMPTZ (NULLable)                                                               |  |
|  | disposition: CaseDisposition (CONFIRMED_FRAUD, FALSE_POSITIVE, LEGITIMATE, SUSPICIOUS_RESOLVED)  |  |
|  | disposition_reason: TEXT (NULLable)                                                             |  |
|  | dispositioned_by: VARCHAR(128) (NULLable)                                                       |  |
|  | dispositioned_at: TIMESTAMPTZ (NULLable)                                                        |  |
|  +-------------------------------------------------------------------------------------------------+  |
|                                                | 1-to-Many                                            |
|                                                v                                                      |
|  +-------------------------------------------------------------------------------------------------+  |
|  |                                            CaseNote                                             |  |
|  |-------------------------------------------------------------------------------------------------|  |
|  | id: UUID (PK)                                                                                   |  |
|  | case_id: UUID (FK -> cases.id, ON DELETE CASCADE)                                               |  |
|  | author_id: VARCHAR(128)                                                                         |  |
|  | author_role: AuditActorType (ANALYST, ADMIN, SYSTEM)                                             |  |
|  | note_type: CaseNoteType (INVESTIGATION, ESCALATION, DISPOSITION, SYSTEM_AUDIT)                  |  |
|  | content: TEXT                                                                                   |  |
|  | created_at: TIMESTAMPTZ                                                                         |  |
|  +-------------------------------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------------------------------+
```

### Lifecycle State Machine

```
               [REVIEW Evaluation / Manual Escalation]
                                 |
                                 v
                            +----------+
                            |   OPEN   |
                            +----------+
                             /        \
                   CLAIM    /          \  ASSIGN (Admin)
                           v            v
                   +---------------+
                   |   IN_REVIEW   |<----------------+
                   +---------------+                 |
                     |     |     ^                   |
            ESCALATE |     |     | REOPEN (Analyst)  | REOPEN (Admin)
                     v     |     |                   |
               +-----------+     |                   |
               | ESCALATED |     |                   |
               +-----------+     |                   |
                     |           |                   |
      RECORD_DISP.   |           | RECORD_DISP.      |
      (Admin only)   |           | (Analyst/Admin)   |
                     v           v                   |
                   +---------------+                 |
                   |   RESOLVED    |-----------------+
                   +---------------+
                           |
                           | CLOSE (Admin)
                           v
                   +---------------+
                   |    CLOSED     |
                   +---------------+
```

---

## 3. Case Management REST API Suite

All endpoints reside under the `/api/v1/cases` router in `backend/app/api/v1/endpoints/cases.py`:

| Method | Path | Summary | Roles Permitted | Concurrency Strategy |
|---|---|---|---|---|
| `GET` | `/api/v1/cases` | Paginated Review Queue with multi-filtering | `ANALYST`, `ADMIN`, `API_CLIENT` | SQL filtering & compact projections |
| `GET` | `/api/v1/cases/summary` | Queue KPI Metrics (Total Open, Unassigned, etc.) | `ANALYST`, `ADMIN`, `API_CLIENT` | Single-query UTC aggregation |
| `GET` | `/api/v1/cases/{case_id}` | Case Investigation Workspace Deep Context | `ANALYST`, `ADMIN`, `API_CLIENT` | `selectinload` relational hydration |
| `POST` | `/api/v1/cases` | Manual Case Escalation | `ANALYST`, `ADMIN` | DB `UNIQUE(transaction_id)` |
| `PATCH` | `/api/v1/cases/{case_id}/assignment` | Mutate Reviewer Assignment (CLAIM/ASSIGN/UNASSIGN) | `ANALYST`, `ADMIN` | Row Lock `SELECT FOR UPDATE` |
| `PATCH` | `/api/v1/cases/{case_id}/status` | Lifecycle Status Mutation (ESCALATE/CLOSE/REOPEN) | `ANALYST`, `ADMIN` | Row Lock `SELECT FOR UPDATE` |
| `GET` | `/api/v1/cases/{case_id}/notes` | Chronological Notes List | `ANALYST`, `ADMIN`, `API_CLIENT` | Chronological ordering |
| `POST` | `/api/v1/cases/{case_id}/notes` | Append Authoritative Investigation Note | `ANALYST`, `ADMIN` | Append-only transaction |
| `POST` | `/api/v1/cases/{case_id}/disposition` | Submit Authoritative Human Disposition | `ANALYST`, `ADMIN` | Row Lock `SELECT FOR UPDATE` |
| `GET` | `/api/v1/cases/{case_id}/timeline` | Authoritative Case Audit Timeline | `ANALYST`, `ADMIN`, `API_CLIENT` | Filtered `AuditEntityType.CASE` query |

---

## 4. Concurrency Hardening & Row-Level Locking

PostgreSQL `SELECT ... FOR UPDATE` is employed across all state mutation workflows in `CaseService`:

1. **Duplicate Case Creation**:
   - `uq_cases_transaction_id` table constraint serializes concurrent attempts.
   - First request returns `HTTP 201 Created`; concurrent losers return `HTTP 409 Conflict`.
2. **Concurrent Reviewer Claims**:
   - Competing analysts attempting `CLAIM` on an `OPEN` case are serialized by row-level lock.
   - First analyst receives `HTTP 200 OK` (`status="IN_REVIEW"`, `assigned_to` set); losing analysts receive `HTTP 409 Conflict` (`"Case '...' is already assigned to '...'"`).
   - Winning analyst retries are idempotent, returning `HTTP 200 OK` without state changes or duplicate audit events.
3. **Concurrent Human Dispositions**:
   - Multiple concurrent disposition requests serialize under row lock.
   - First request transitions case from active to `RESOLVED` (`HTTP 200 OK`).
   - Subsequent requests evaluate terminal state and return `HTTP 422 Unprocessable Entity` with zero deadlocks and zero `500` server errors.
   - Exactly one `DISPOSITION` note and one `CASE_DISPOSITION_RECORDED` audit log are persisted in strict chronological sequence.

---

## 5. RBAC Security Matrix & Production Fail-Closed Authentication

The system enforces role-based access control via `ActorContext` and `require_role()`:

| Operation | `ANALYST` | `ADMIN` | `API_CLIENT` | `SYSTEM` |
|---|:---:|:---:|:---:|:---:|
| **Queue & Summary Read** | ✅ | ✅ | ✅ | ✅ |
| **Case Detail & Notes Read** | ✅ | ✅ | ✅ | ✅ |
| **Manual Escalation** | ✅ | ✅ | ❌ (403) | ✅ |
| **Self-Claim (CLAIM)** | ✅ | ✅ | ❌ (403) | ❌ |
| **Supervisor Assignment (ASSIGN)** | ❌ (403) | ✅ | ❌ (403) | ❌ |
| **Self-Unassign (UNASSIGN)** | ✅ (Own) | ✅ (Any) | ❌ (403) | ❌ |
| **Escalate Case (ESCALATE)** | ✅ | ✅ | ❌ (403) | ❌ |
| **Disposition Active Case** | ✅ | ✅ | ❌ (403) | ❌ |
| **Disposition Escalated Case** | ❌ (403) | ✅ | ❌ (403) | ❌ |
| **Reopen Resolved Case** | ✅ | ✅ | ❌ (403) | ❌ |
| **Reopen Closed/Archived Case** | ❌ (403) | ✅ | ❌ (403) | ❌ |
| **Close & Archive Case** | ✅ | ✅ | ❌ (403) | ❌ |
| **Append Investigation Note** | ✅ | ✅ | ❌ (403) | ❌ |
| **Query Case Audit Timeline** | ✅ | ✅ | ✅ | ✅ |

### Production Fail-Closed Policy
When `ALLOW_DEV_ACTOR_HEADERS=False`, development header injection (`X-Actor-ID`, `X-Actor-Role`) is rejected with `HTTP 401 Unauthorized`.

---

## 6. Frontend Review Queue & Case Investigation Workspace

The React 18 + TypeScript frontend (`frontend/src/`) provides:

1. **Review Queue View (`ReviewQueue.tsx`)**:
   - 7 Operational KPI Summary Cards: Total Open, Unassigned, In Review, Escalated, Resolved Today, Resolved 24h, Critical Priority.
   - Multi-dimensional filtering: Lifecycle status, Priority, Assignee (including 'unassigned'), Risk Tier, Risk Score Range (0–100), Search term (Case #, Account ID, Merchant), and Date range.
   - Server-side sorting and pagination.
2. **Case Investigation Workspace (`CaseInvestigationWorkspace.tsx`)**:
   - **Left Column**: Point-in-time financial transaction metadata, complete 55-feature snapshot viewer with category filters.
   - **Center Column**: Calibrated risk score gauge, decision policy badges, plain-English reason codes, triggered rule matches, and interactive `ShapWaterfall` feature attribution component.
   - **Right Column**: Reviewer assignment controls, lifecycle action toolbar, append-only notes composer with author badges, and authoritative case audit timeline.
3. **Action Modals**:
   - `CreateCaseModal`, `DispositionModal`, `AssignModal`, `EscalateModal`, `CloseModal`, `ReopenModal`.
4. **Development Actor Context Switcher (`ActorContextSwitcher.tsx`)**:
   - Development convenience tool hidden and disabled in production mode (`ALLOW_DEV_ACTOR_HEADERS=false`).

---

## 7. Downstream Ground-Truth Data Contract

Resolved cases provide verified, point-in-time ground-truth labels for downstream Phase 14 retraining loops:
- `Transaction.features_snapshot`: Complete 55 raw and engineered features captured at transaction ingestion.
- `RiskEvaluation.model_score`: Uncalibrated model output $[0.0, 1.0]$.
- `RiskEvaluation.risk_score`: Calibrated integer risk score $[0, 100]$.
- `Case.disposition`: Authoritative human label (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `LEGITIMATE`, `SUSPICIOUS_RESOLVED`).
- `Case.disposition_reason`: Human investigator rationale ($\ge 10$ characters).
- `Case.dispositioned_at`: UTC timestamp of resolution.

---

## 8. Verification Results

### Backend Test Suite
- **Baseline Tests (Phases 1–12.4)**: 906 passed
- **Phase 12.5 E2E & Concurrency Tests Added**: 6 passed
  - `test_full_multi_turn_case_lifecycle_e2e`: PASSED
  - `test_concurrent_duplicate_case_creation`: PASSED
  - `test_concurrent_reviewer_claims`: PASSED
  - `test_concurrent_dispositions`: PASSED
  - `test_case_rbac_matrix_e2e`: PASSED
  - `test_ground_truth_disposition_data_contract`: PASSED
- **Total Backend Tests**: **912 passed**, 0 failures, 14 warnings.

### Frontend Production Build
- Command: `npm run build --prefix frontend`
- TypeScript Errors: **0**
- Build Status: Clean Vite production build.

---

## 9. Architectural Decision Record: ADR-017

### ADR-017: Human Review & Case Management Lifecycle Architecture
- **Status:** Approved
- **Context:** Automated fraud detection models generate `REVIEW` tier decisions requiring expert human investigation, structured disposition recording, and tamper-proof audit trails.
- **Decision:**
  1. Enforce strict 1-to-1 cardinality between financial transactions and investigation cases via database constraint `UNIQUE(transaction_id)`.
  2. Implement an explicit 5-state lifecycle state machine (`OPEN`, `IN_REVIEW`, `ESCALATED`, `RESOLVED`, `CLOSED`) with row-level locks (`SELECT FOR UPDATE`).
  3. Decouple assignment actions (`CLAIM`, `ASSIGN`, `UNASSIGN`) from terminal states, ensuring assignment state machine integrity.
  4. Require mandatory explanatory rationale ($\ge 10$ characters) for all lifecycle mutations and human review dispositions.
  5. Isolate `AuditEntityType.CASE` events for authoritative case timeline reconstruction.
  6. Persist immutable point-in-time feature snapshots and human dispositions to serve as ground-truth training datasets for Phase 14 retraining pipelines.
- **Consequences:** Provides concurrency-safe human review operations, prevents duplicate cases, guarantees full audit compliance, and provides high-fidelity training data for continuous model improvement.
