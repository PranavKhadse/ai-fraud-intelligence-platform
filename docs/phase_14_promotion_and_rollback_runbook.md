# Phase 14.6: Staged Model Promotion & Rollback Operational Runbook

## Executive Summary & Core Governance Principles

This operational runbook governs the staged model promotion, rollback, and state reconciliation procedures for the **AI-Powered Fraud Detection & Risk Intelligence Platform**.

### Mandatory Invariants:
1. **Explicit / Manual Execution**: Model promotions and rollbacks are strictly manual, authenticated operations. There is **zero automatic promotion** in the platform.
2. **Deterministic Governance Binding**: Promotion requires an explicit `APPROVED` human sign-off cryptographically bound to the exact candidate artifact checksums, active Champion checksums, Phase 14.4 comparison evidence, and Phase 14.5 eligibility assessment.
3. **No Automatic Retraining**: Rollback restores previously verified, immutable historical bundles without altering hyperparameters or re-optimizing thresholds.
4. **Authoritative PostgreSQL State**: PostgreSQL is the single source of truth for active Champion status. Exactly one active Champion is permitted by database constraints and service validation.
5. **No Automatic Database Rollback Post-Commit**: If post-commit verification encounters an anomaly, the subsystem records a critical `RECOVERY_REQUIRED` operational state and halts rather than performing hidden automatic mutations.

---

## 1. Promotion Prerequisites & Precondition Matrix

A candidate model (e.g. `v1.1.0`) must satisfy all 16 precondition checks before promotion execution begins:

| # | Check Dimension | Validation Requirement | Failure Exception |
|---|---|---|---|
| 1 | **Database Registration** | Candidate exists in PostgreSQL `model_registry_entries` | `PromotionPreconditionError` |
| 2 | **Bundle On-Disk Completeness** | `model.joblib`, `preprocessor.joblib`, and `manifest.json` exist | `PromotionPreconditionError` |
| 3 | **Candidate Lifecycle Status** | Status is `CANDIDATE` or `CHALLENGER` (`REJECTED`/`ARCHIVED` blocked) | `PromotionPreconditionError` |
| 4 | **Active Champion Exclusivity** | Candidate is not already marked `is_active_champion = True` | `PromotionPreconditionError` |
| 5 | **Active Champion Integrity** | Exactly 1 active Champion exists with status `CHAMPION` | `PromotionPreconditionError` |
| 6 | **Governance Eligibility** | Phase 14.5 assessment confirms `is_eligible = True` (11/11 gates) | `PromotionPreconditionError` |
| 7 | **Sign-Off Decision** | Explicit `SignOffDecision.APPROVED` human sign-off record | `PromotionPreconditionError` |
| 8 | **Signer RBAC** | Signer role is strictly `ADMIN` or `ANALYST` | `PromotionPreconditionError` |
| 9 | **Executing Actor RBAC** | Executing actor role is strictly `ADMIN` or `ANALYST` | `PromotionPreconditionError` |
| 10 | **Sign-Off Business Rationale** | Mandatory rationale $\ge 15$ non-whitespace characters | `PromotionPreconditionError` |
| 11 | **Version Linkage** | Sign-off `candidate_version` and `champion_version` match DB state | `PromotionPreconditionError` |
| 12 | **Assessment SHA-256 Linkage** | Sign-off `eligibility_assessment_sha256` matches assessment JSON | `PromotionIntegrityError` |
| 13 | **Comparison Evidence SHA-256** | Comparison SHA-256 matches primary Phase 14.4 artifact | `PromotionIntegrityError` |
| 14 | **Candidate Checksums** | On-disk candidate binary SHA-256 hashes match DB records | `PromotionIntegrityError` |
| 15 | **Champion Checksums** | Active filesystem Champion SHA-256 hashes match DB records | `PromotionIntegrityError` |
| 16 | **Decision Threshold & Schema** | Operating threshold $\tau^* \in (0.0, 1.0)$, 55 canonical features valid | `PromotionPreconditionError` |

---

## 2. Promotion Operation Lifecycle & State Machine

Every promotion operation executes under a durable, on-disk journal tracking each stage:

```
REQUESTED
    │
    ▼ (Preconditions verified & SELECT FOR UPDATE acquired)
VALIDATED
    │
    ▼ (Isolated staging 55-feature inference & metadata contract prepared)
STAGED
    │
    ▼ (Historical bundle archived & active filesystem artifacts swapped)
FILESYSTEM_SWAPPED
    │
    ▼ (Atomic DB transaction & PostgreSQL AuditLog committed)
DB_COMMITTED
    │
    ▼ (Post-commit consistency check verified: DB == filesystem == hashes)
FINALIZED
```

### Failure & Compensation States:
- **`FAILED_BEFORE_COMMIT`**: Any failure during staging, synthetic inference, or filesystem swap automatically restores active artifacts from `.active_champion_backup_{operation_id}`, rolls back the DB transaction, and cleans staging.
- **`RECOVERY_REQUIRED`**: If post-commit consistency verification fails, DB state is preserved as authoritative and a critical operator alert is logged for manual reconciliation.
- **`RECOVERY_COMPLETED`**: Startup recovery successfully reconciled filesystem artifacts to match PostgreSQL active Champion.

---

## 3. Staged Promotion Execution Protocol

### Step-by-Step Procedure:
1. **Concurrency Lock**: Begin PostgreSQL transaction with `SELECT FOR UPDATE` on active Champion and candidate rows.
2. **Re-Read & Revalidate**: Verify versions, statuses, and hashes have not been mutated by concurrent processes.
3. **Journal Initialization**: Write `ml/models/registry/operations/{operation_id}.json` with state `REQUESTED` $\to$ `VALIDATED`.
4. **Isolated Staging Validation**:
   - Copy candidate bundle to `ml/models/registry/staging/{operation_id}/`.
   - Verify staging file checksums.
   - Deserialize `model.joblib` and `preprocessor.joblib`.
   - Run synthetic 55-feature matrix inference (`create_synthetic_55_feature_dataframe`).
   - Validate operating threshold $\tau^*$ and generate runtime `model_metadata.json` contract.
5. **Historical Archive of Current Champion**:
   - If `ml/models/registry/bundles/{champion_version}/` does not exist, copy active artifacts and generate `manifest.json` before replacing active files.
6. **Active Artifact Backup**:
   - Copy current active artifacts to `staging/.active_champion_backup_{operation_id}`.
7. **Filesystem Swap**:
   - Copy candidate model, preprocessor, and formatted `model_metadata.json` to `ml/models/artifacts/`.
   - Verify on-disk checksums. Transition journal to `FILESYSTEM_SWAPPED`.
8. **Atomic DB Update & Audit Log**:
   - Demote previous Champion: `status = ARCHIVED`, `is_active_champion = False`.
   - Promote Candidate: `status = CHAMPION`, `is_active_champion = True`, `promoted_at = UTC NOW`, `promoted_by = actor_id`, `promotion_rationale = rationale`.
   - Insert `AuditLog` entry (`action = "MODEL_PROMOTION"`, `event_type = "MODEL_PROMOTION_EXECUTED"`).
   - Commit transaction. Transition journal to `DB_COMMITTED`.
9. **Post-Commit Verification**:
   - Verify DB active champion equals candidate version.
   - Verify active filesystem artifact checksums match candidate checksums.
10. **Finalization & Audit Execution Record**:
    - Transition journal to `FINALIZED`.
    - Persist immutable `PromotionExecutionRecord` to `ml/models/registry/promotions/promotion_{candidate_version}_{timestamp}.json`.
    - Clean temporary staging and backup directories.

---

## 4. Model Rollback Protocol

Rollback restores a previously verified, archived historical model bundle (e.g. `v1.0.0`) to active Champion status.

### Rollback Procedure:
1. **Target Bundle Verification**:
   - Verify `ml/models/registry/bundles/{target_version}/` exists with `model.joblib`, `preprocessor.joblib`, and `manifest.json`.
   - Verify SHA-256 hashes against PostgreSQL target entry.
2. **Actor Authorization**: Executing actor must possess `ADMIN` or `ANALYST` role with business rationale $\ge 15$ characters.
3. **Isolated Staging Validation**: Load target bundle in staging, run synthetic 55-feature inference, and construct runtime `model_metadata.json`.
4. **Row Locks**: Acquire `SELECT FOR UPDATE` on active Champion and target version.
5. **Active Backup**: Backup current active artifacts to `staging/.active_champion_backup_{operation_id}`.
6. **Filesystem Swap**: Copy target bundle files to `ml/models/artifacts/`.
7. **Atomic DB Transition**:
   - Current Champion: `status = ROLLED_BACK`, `is_active_champion = False`, `rolled_back_at = UTC NOW`, `rolled_back_by = actor_id`, `rollback_rationale = rationale`.
   - Target Model: `status = CHAMPION`, `is_active_champion = True`.
   - Insert `AuditLog` entry (`action = "MODEL_ROLLBACK"`, `event_type = "MODEL_ROLLBACK_EXECUTED"`).
   - Commit transaction.
8. **Post-Commit Consistency Verification**: Re-verify DB active champion == target version == filesystem artifacts.
9. **Execution Record**: Write immutable `RollbackExecutionRecord` to `ml/models/registry/rollbacks/rollback_{target_version}_{timestamp}.json`.

---

## 5. Crash Recovery & Startup State Reconciliation

The `PromotionRecoveryEngine` automatically reconciles state on application startup:

```python
from ml.lifecycle.promotion import PromotionRecoveryEngine

recovery = PromotionRecoveryEngine()
result = await recovery.reconcile_active_champion_state(session=db_session)
```

### Reconciliation Rules:
1. **Crash during `FILESYSTEM_SWAPPED` (Pre-Commit)**:
   - Database transaction did not commit; DB active Champion is authoritative.
   - Filesystem active artifacts are restored from historical bundle of DB active Champion.
   - Journal marked `RECOVERY_COMPLETED`.
2. **Crash during `DB_COMMITTED` (Post-Commit)**:
   - Database transaction committed; DB new Champion is authoritative.
   - Filesystem active artifacts verified and finalization completed.
   - Journal marked `FINALIZED`.
3. **State `RECOVERY_REQUIRED`**:
   - Emits critical error requiring manual inspection.

---

## 6. Audit Trail & Lineage Schema

Each promotion and rollback generates immutable cryptographic audit records:

- **Promotion Execution Record**: `ml/models/registry/promotions/promotion_{version}_{timestamp}.json`
- **Rollback Execution Record**: `ml/models/registry/rollbacks/rollback_{version}_{timestamp}.json`
- **PostgreSQL Audit Log**: `audit_logs` table (`event_type = 'MODEL_PROMOTION_EXECUTED'` or `'MODEL_ROLLBACK_EXECUTED'`)
- **Operation Journal**: `ml/models/registry/operations/{operation_id}.json`
