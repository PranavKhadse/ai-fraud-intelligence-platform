# PROJECT_STATUS.md — Project Tracking & Status Dashboard

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform
> **Last Updated:** Current Date (Phase 6 Implementation Completed)
> **Current Active Phase:** **Phase 6 — Risk Engine & Decision Framework (Completed)**

---

## 📊 Phase Progress Summary

| Phase | Description | Status | Target Completion |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Project Foundation & Architecture** | 🟢 **Completed** | Milestone 0 |
| **Phase 1** | **Data Pipeline & Ingestion** | 🟢 **Completed** | Milestone 1 |
| **Phase 2** | **Exploratory Data Analysis (EDA) & Insights** | 🟢 **Completed** | Milestone 2 |
| **Phase 3** | **Behavioral Feature Engineering** | 🟢 **Completed** | Milestone 3 |
| **Phase 4** | **Baseline & Advanced ML Models** | 🟢 **Completed** | Milestone 4 |
| **Phase 5** | **Imbalance Handling & Cost Optimization** | 🟢 **Completed** | Milestone 5 |
| **Phase 6** | **Risk Engine & Decision Framework** | 🟢 **Completed** | Milestone 6 (Current) |
| **Phase 7** | **Explainability & Reason Codes** | ⚪ Pending | Next Milestone |
| **Phase 8** | **Fraud Detection API (FastAPI)** | ⚪ Pending | Phase 8 |
| **Phase 9** | **Database & Persistence (PostgreSQL)** | ⚪ Pending | Phase 9 |
| **Phase 10** | **Real-Time Detection & Benchmarking** | ⚪ Pending | Phase 10 |
| **Phase 11** | **Fraud Intelligence Dashboard** | ⚪ Pending | Phase 11 |
| **Phase 12** | **Human Review & Case Management** | ⚪ Pending | Phase 12 |
| **Phase 13** | **ML & Model Monitoring** | ⚪ Pending | Phase 13 |
| **Phase 14** | **MLOps, Retraining & Model Registry** | ⚪ Pending | Phase 14 |
| **Phase 15** | **Security, Auth & Audit Logging** | ⚪ Pending | Phase 15 |
| **Phase 16** | **Containerization & Deployment** | ⚪ Pending | Phase 16 |
| **Phase 17** | **Comprehensive Testing Suite** | ⚪ Pending | Phase 17 |
| **Phase 18** | **Final Portfolio Polish & Runbooks** | ⚪ Pending | Phase 18 |

---

## ✅ Phase 4 Deliverable Checklist (Baseline & Advanced ML Models)

- [x] Trained 4 model architectures (Logistic Regression, Random Forest, XGBoost, LightGBM) on 1,296,675 training records using 55 deterministic features.
- [x] Enforced cross-library process isolation on Windows to prevent OpenMP memory collisions between XGBoost (`vcomp140.dll`) and LightGBM (`libgomp-1.dll`).
- [x] Selected **XGBoost Classifier** as the champion model architecture based on superior Validation PR-AUC ($0.96190$) and ROC-AUC ($0.99926$).
- [x] Identified Phase 4 F1-optimal threshold ($\tau = 0.94$) achieving Precision = 93.72%, Recall = 89.19%, and F1 = 0.91397 on Validation.
- [x] Persisted champion artifacts (`champion_model.joblib`, `champion_preprocessor.joblib`, `model_metadata.json`, `threshold_analysis.json`).
- [x] Authored comprehensive documentation in `docs/model_training_report.md`.

---

## ✅ Phase 5 Deliverable Checklist (Imbalance Handling & Cost Optimization)

- [x] Architected modular, configurable cost-optimization package under `ml/cost_optimization/` (`config`, `cost_engine`, `optimize`, `sensitivity`, `oot_evaluation`).
- [x] Implemented typed, validated `CostConfig` dataclass supporting $C_{\text{FP}}$, $C_{\text{FN}}$, $C_{\text{review}}$, $C_{\text{TN}}$, $C_{\text{TP}}$ with illustrative business assumptions ($C_{\text{FP}}=\$15.00, C_{\text{FN}}=\$200.00, C_{\text{review}}=\$5.00$).
- [x] Conducted deterministic validation threshold sweep ($0.01$ to $0.99$, step $0.01$) discovering minimum-cost threshold $\tau^* = 0.78$ on the Validation partition.
- [x] Proven that $\tau^* = 0.78$ cuts validation expected cost from $\$27,495.00$ ($\tau=0.94$) to $\$16,910.00$ (38.5% cost reduction), capturing 71 additional fraud cases.
- [x] Conducted sensitivity analysis across discrete scenarios (Scenario A: Customer-friendly, Scenario B: Balanced, Scenario C: Fraud-loss-sensitive) and 49-cell 2D cost surface ($C_{\text{FP}} \times C_{\text{FN}}$).
- [x] Authored Probability Calibration Feasibility & Leakage Governance Report (`docs/calibration_feasibility_report.md`), demonstrating empirical score distortion from `scale_pos_weight` and proving ranking-optimality of decision thresholding without retraining.
- [x] Executed final frozen Out-of-Time (OOT) evaluation on protected holdout (`test_features.parquet`, 277,860 rows):
  - $\tau = 0.78$ delivered **$14,245.00** total cost vs **$20,890.00** at $\tau = 0.94$ (**31.81% cost reduction**).
  - Detected **871 / 924 frauds (94.26% recall)** vs 823 / 924 (89.07% recall) at $\tau=0.94$ (+48 additional fraud cases caught).
  - Reduced cost per fraud detected from **$25.38** to **$16.35**.
  - Verified SHA-256 artifact immutability before and after OOT execution.
- [x] Generated high-resolution diagnostic plots: `docs/figures/cost_vs_threshold.png`, `docs/figures/cost_sensitivity_heatmap.png`, `docs/figures/oot_threshold_comparison.png`.
- [x] Authored authoritative reports in `docs/cost_optimization_report.md` and `docs/oot_evaluation_report.md`.
- [x] Passed 82 / 82 automated ML tests across `tests/ml/`.

---

## ✅ Phase 6 Deliverable Checklist (Risk Engine & Decision Framework)

- [x] **Increment 1: Core Normalization & Tier Mapping**:
  - Implemented exact scalar and vectorized model score normalization ($[0.0, 1.0] \to [0, 100]$) in `ml/risk_engine/normalization.py`.
  - Implemented monotonic 4-band risk tier mapping (`LOW`: 0–34, `MEDIUM`: 35–59, `HIGH`: 60–77, `CRITICAL`: 78–100).
- [x] **Increment 2: Decision Policy Engine & End-to-End Evaluator**:
  - Implemented `DecisionPolicyConfig`, `DecisionResult`, and `DecisionPolicyEngine` supporting `TRI_TIER` ($\tau_{\text{rev}}=0.35, \tau_{\text{blk}}=0.78$) and `BINARY_AUTO` ($\tau_{\text{blk}}=0.78$) in `ml/risk_engine/policy.py`.
  - Built production `RiskEvaluator` in `ml/risk_engine/evaluator.py` loading frozen champion artifacts in read-only mode.
- [x] **Increment 3: Provenance Hardening & Metadata Tolerance**:
  - Implemented `BatchDecisionSummary` aggregation with deep immutability.
  - Added model version provenance tracking from `model_metadata.json` and robust non-predictive metadata column passthrough.
- [x] **Increment 4: Immutable Policy Reason Codes**:
  - Defined typed `DecisionReasonCode` enum and deterministic decision boundary explanations.
- [x] **Increment 5A & 5B: Deterministic Rule Engine & Hybrid Precedence**:
  - Implemented immutable `RiskRule`, `RuleMatch`, and `RuleEngine` in `ml/risk_engine/rules.py` supporting typed operators (`>`, `<`, `in`, `is_true`).
  - Integrated hybrid decision coordinator with deterministic precedence ($\text{BLOCK} \succ \text{REVIEW} \succ \text{APPROVE} \succ \text{MONITOR}$) and non-downgrading protected model blocks.
- [x] **Increment 6: Standard Rule Catalog, Empirical Benchmark & Reconciliation Audit**:
  - Implemented verified 6-rule standard catalog in `ml/risk_engine/catalog.py` (2 `REVIEW` rules, 4 `MONITOR` rules, 0 hard-block rules).
  - Completed strictly read-only empirical benchmark and independent reconciliation audit on $555,719$ validation and OOT transactions.
  - Proved 100% strict-blocking invariance ($\Delta = 0$ TP, FP, FN, TN, precision, recall, F1).
  - Clarified review-queue dynamics: catalog acts as a safety-net manual review escalation layer, diluting review queue purity ($4.70\% \to 1.93\%$ in OOT) because ML already captures $98.7\%$ of high-zscore frauds in the hard block tier.
  - Authored authoritative Phase 6 report in `docs/risk_engine_report.md`.
  - Expanded test suite to **250 risk engine tests** and **332 total ML tests** with 100% pass rate.

---

## 🏛 Architectural Decision Records (ADRs)

### ADR-001: Modular Monorepo Scaffolding
- **Status**: Accepted.

### ADR-002: Lightweight Phase-by-Phase Dependency Management
- **Status**: Accepted.

### ADR-003: Measurable Non-Functional Requirements (Latency & Throughput)
- **Status**: Accepted.

### ADR-004: Calibrated 0–100 Risk Scoring with Tri-Tier Decision Policy
- **Status**: Accepted.

### ADR-005: Time-Aware Out-of-Time (OOT) Splitting & Zero-Leakage Protocol
- **Status**: Accepted (Phase 1).

### ADR-006: Canonical 15-Field Transaction Schema & Columnar Parquet Storage
- **Status**: Accepted (Phase 1).

### ADR-007: Empirical Fraud Domain Insights & Prioritized Feature Blueprint
- **Status**: Accepted (Phase 2).

### ADR-008: Strict Point-in-Time Behavioral Feature Engineering & Zero-Leakage Protocol
- **Status**: Accepted (Phase 3).

### ADR-009: Process-Isolated Multi-Model Training & Champion Selection
- **Status**: Accepted (Phase 4).

### ADR-010: Cost-Sensitive Decision Boundary Optimization & Leakage-Safe Governance
- **Status**: Accepted (Phase 5).

### ADR-011: Hybrid Risk Decisioning & Standard Rule Catalog Governance
- **Context**: Business domain rules and compliance constraints must operate alongside ML predictions without overriding high-confidence ML blocks or creating catastrophic false-positive spikes.
- **Decision**: Structure the standard rule catalog as a risk-intelligence and manual-review escalation layer (containing `REVIEW` and `MONITOR` rules, with zero automated hard-block rules). Enforce strict precedence where ML `BLOCK` outcomes are immutable and cannot be downgraded by lower-tier rules. Relegate geographic speed anomalies to `MONITOR` status to avoid false-positive disruptions on e-commerce transactions.
- **Status**: Accepted (Phase 6).

---

## ⚠️ Known Constraints & Risk Register

1. **Review Queue Operational Sizing**: Hybrid rule overrides add $\approx 1,039$ transactions to the OOT manual review queue ($+168.4\%$ volume increase). Real-world deployments must ensure sufficient analyst capacity to maintain SLA targets.
2. **Static Fixed-Cost Assumptions**: Current benchmark evaluations apply fixed unit costs ($C_{\text{FP}}=\$15, C_{\text{FN}}=\$200, C_{\text{REV}}=\$5$). Dynamic amount-weighted risk scoring is recommended for future financial optimization.

---

## ⏭ Next Step: Preparation for Phase 7 (Explainability & Reason Codes)

When approved to start Phase 7:
- Integrate TreeSHAP feature attributions on top of the XGBoost champion model.
- Map local SHAP attribution values into standardized human-readable reason codes for all non-approved transactions.
- Implement waterfall visualization exports for downstream analyst case management.
