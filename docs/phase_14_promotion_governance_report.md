# Phase 14.5: Model Promotion Gate & Human Sign-Off Governance Report

## Executive Summary

Phase 14.5 establishes the formal **Promotion Gate & Human Sign-Off Governance System** for the AI-Powered Fraud Detection & Risk Intelligence Platform. This subsystem provides a deterministic, multi-dimensional governance framework to evaluate whether a trained and evaluated candidate model meets all enterprise risk, statistical, operational, latency, integrity, and safety requirements before any production transition can occur.

In accordance with strict enterprise governance principles:
- **No Automatic Promotion**: Evaluation of promotion gates determines *eligibility only*.
- **Mandatory Human Sign-Off**: Transition to production requires explicit, authenticated human approval with RBAC role authorization (`ADMIN` or `ANALYST`) and a recorded rationale of $\ge 15$ characters.
- **Champion Immutability**: The active Champion (v1.0.0) remains active, and all Champion artifacts are dynamically verified for cryptographic immutability (SHA-256).
- **Candidate Invariant Preservation**: Candidate v1.1.0 remains in lifecycle status `CANDIDATE` with `is_active_champion = False`.
- **Neutral Terminology**: All performance differentials are reported using strictly mathematical, neutral deltas (`POSITIVE`, `NEGATIVE`, `ZERO`, absolute deltas, percentage-point deltas, and relative percentage changes).

---

## 1. Governance Gate Specification & Categorization

The governance engine evaluates **11 explicit criteria** across 6 operational dimensions. Every criterion is explicitly documented as either an **Existing System Requirement** (established in foundational phases) or a **Governance Policy Configuration** (enterprise policy thresholds defined for promotion gating).

| Gate ID | Category | Target Metric | Operator | Threshold / Constraint | Policy Category |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `gate_oot_pr_auc_parity` | Statistical Ranking | OOT PR-AUC Parity | $\ge$ | $\text{Champion} - 0.0050$ | Governance Policy Configuration |
| `gate_val_pr_auc_parity` | Statistical Ranking | Validation PR-AUC Parity | $\ge$ | $\text{Champion} - 0.0050$ | Governance Policy Configuration |
| `gate_oot_recall_non_regression` | Detection Effectiveness | OOT Recall (Catch Rate) | $\ge$ | $\text{Champion} + 0.0000$ | Governance Policy Configuration |
| `gate_oot_fn_ceiling` | Detection Effectiveness | OOT Missed Fraud (FN) | $\le$ | $\text{Champion} + 0$ | Governance Policy Configuration |
| `gate_oot_expected_cost` | Financial Risk | OOT Expected Decision Cost | $\le$ | $\text{Champion} + \$0.00$ | Governance Policy Configuration |
| `gate_oot_fpr_ceiling` | Operational Stability | OOT False Positive Rate | $\le$ | $0.0020$ ($0.20\%$) | Governance Policy Configuration |
| `gate_oot_block_precision` | Operational Stability | OOT Block Action Precision | $\ge$ | $0.7000$ ($70.0\%$) | Governance Policy Configuration |
| `gate_inference_latency_p95` | Latency SLA | p95 Inference Latency | $\le$ | $\min(5.0\text{ ms}, 1.50 \times \text{Champion})$ | Governance Policy Configuration |
| `gate_champion_artifact_integrity` | Model Integrity | Champion Dynamic Checksum | $==$ | Registered SHA-256 Hashes | Existing System Requirement |
| `gate_feature_schema_compatibility` | Schema Contract | Predictive Feature Count | $==$ | $55$ Canonical Features | Existing System Requirement |
| `gate_evaluation_partition_integrity` | Partition Integrity | Validation & OOT Sample Counts | $==$ | $\text{Val}=277,859, \text{OOT}=277,860$ | Existing System Requirement |

---

## 2. Gate Evaluation Results for Candidate v1.1.0

The `PromotionGateEvaluator` evaluated Candidate v1.1.0 using the verified Phase 14.4 comparative evaluation artifact (`ml/models/registry/comparisons/comparison_v1.0.0_vs_v1.1.0.json`) and dynamic on-disk checksum calculations.

### Summary Evaluation Metric Table

| Metric / Dimension | Champion (v1.0.0) | Candidate (v1.1.0) | Mathematical Delta | Gate Threshold | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Operating Threshold ($\tau$)** | $0.9400$ | $0.7800$ | $-0.1600$ | N/A | Evaluated |
| **OOT PR-AUC** | $0.96054$ | $0.96054$ | $0.00000$ | $\ge 0.95554$ | **PASS** |
| **Validation PR-AUC** | $0.96190$ | $0.96190$ | $0.00000$ | $\ge 0.95690$ | **PASS** |
| **OOT Fraud Catch Rate (Recall)** | $0.89069$ ($823/924$) | $0.94264$ ($871/924$) | $+5.195\text{ pp}$ ($+48\text{ frauds}$) | $\ge 0.89069$ | **PASS** |
| **OOT Missed Fraud (FN)** | $101$ | $53$ | $-48\text{ missed frauds}$ | $\le 101$ | **PASS** |
| **OOT Expected Decision Cost** ($C_{\text{FP}}=\$15, C_{\text{FN}}=\$200$) | $\$20,890.00$ | $\$14,245.00$ | $-\$6,645.00$ ($-31.81\%$) | $\le \$20,890.00$ | **PASS** |
| **OOT False Positive Rate (FPR)** | $0.00025$ ($69/276,936$) | $0.00088$ ($243/276,936$) | $+0.063\text{ pp}$ | $\le 0.00200$ | **PASS** |
| **OOT Block Precision** | $0.92265$ ($823/892$) | $0.78187$ ($871/1,114$) | $-14.078\text{ pp}$ | $\ge 0.70000$ | **PASS** |
| **Inference Latency (p95)** | $0.976\text{ ms}$ | $1.037\text{ ms}$ | $+0.061\text{ ms}$ | $\le 1.464\text{ ms}$ | **PASS** |
| **Champion Artifact SHA-256** | Validated on Disk | Validated on Disk | Matches Registered Metadata | Exact Match | **PASS** |
| **Feature Schema Dimension** | $55$ Features | $55$ Features | $0$ Mismatches | Exactly $55$ | **PASS** |
| **Validation / OOT Row Counts** | $277,859 / 277,860$ | $277,859 / 277,860$ | $0$ Discrepancies | Exactly Matched | **PASS** |

### Overall Eligibility Result
- **Total Gates Evaluated**: 11
- **Passed Gates**: 11
- **Failed Gates**: 0
- **Blocked Gates**: 0
- **Promotion Eligibility (`is_eligible`)**: **`True`**

---

## 3. Dynamic Artifact Integrity Verification

Champion artifact hashes were calculated dynamically on disk and compared against the authoritative registered metadata:

| Artifact Path | Computed SHA-256 Checksum | Registered Metadata Hash | Integrity Status |
| :--- | :--- | :--- | :---: |
| `ml/models/artifacts/champion_model.joblib` | `5598fc3c2243d54c7d0d0f4d45d8b76ceaf0f1ba8e59ea2c0836109dc0e32b49` | `5598fc3c2243d54c7d0d0f4d45d8b76ceaf0f1ba8e59ea2c0836109dc0e32b49` | **VERIFIED** |
| `ml/models/artifacts/champion_preprocessor.joblib` | `24f4783c50bdf922e3919e992b1b590e80e30379895085e783c921fe7c164a63` | `24f4783c50bdf922e3919e992b1b590e80e30379895085e783c921fe7c164a63` | **VERIFIED** |
| `ml/models/artifacts/model_metadata.json` | `deb0f0e8f7dc93f0b2f7d3eb81a176883bfd1279148d2fc7f7ba523be932c021` | `deb0f0e8f7dc93f0b2f7d3eb81a176883bfd1279148d2fc7f7ba523be932c021` | **VERIFIED** |

---

## 4. Human Sign-Off Engine & RBAC Architecture

The `HumanSignOffEngine` manages formal sign-off records with strict auditability and non-repudiation:
1. **Role-Based Access Control**:
   - Authorized Roles: `ADMIN`, `ANALYST`
   - Unauthorized Roles (e.g., `VIEWER`, `GUEST`) are strictly rejected with `SignOffAuthorizationError`.
2. **Precondition Enforcement**:
   - An `APPROVED` decision requires `is_eligible == True`. Attempting to approve an ineligible candidate raises `SignOffPreconditionError`.
   - A `REJECTED` decision is allowed regardless of eligibility.
3. **Audit Trail & Non-Repudiation**:
   - `rationale` is mandatory with a minimum length of 15 non-whitespace characters.
   - Every sign-off records `signed_off_at` (UTC timestamp), `decision`, `signer_id`, `signer_role`, `assessment_id`, and `evidence_artifact_sha256`.
4. **Conflict Prevention & Immutability**:
   - Attempting to overwrite an existing sign-off record without explicit authorization raises `SignOffConflictError`.
   - Test suites execute against isolated fixtures (`tmp_path`) to ensure zero fake records pollute the production directory (`ml/models/registry/signoffs/`).

---

## 5. Model Registry State Invariants

Verification of PostgreSQL `model_registry_entries` and on-disk bundle manifests confirms complete preservation of model lifecycle states:

```
[Registry State Verification]
Champion (v1.0.0):
  status             = CHAMPION
  is_active_champion = True
  promoted_at        = 2026-03-08 00:00:00+00
  promoted_by        = system_init

Candidate (v1.1.0):
  status             = CANDIDATE
  is_active_champion = False
  promoted_at        = NULL
  promoted_by        = NULL
```

---

## 6. Conclusion & Readiness

Phase 14.5 is fully implemented, verified, and integrated into `ml.lifecycle`.
Candidate v1.1.0 has satisfied all 11 promotion gates and is marked **ELIGIBLE FOR PROMOTION** (`is_eligible = True`).

The system is now fully prepared for **Phase 14.6: Model Promotion, Deployment, and Rollback Execution**, which will perform the atomic activation of Candidate v1.1.0 upon formal authorized sign-off.
