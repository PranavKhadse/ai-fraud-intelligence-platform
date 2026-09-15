# PROJECT_STATUS.md — Project Tracking & Status Dashboard

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform
> **Last Updated:** Current Date (Phase 8 Implementation Completed)
> **Current Active Phase:** **Phase 8 — Fraud Detection API (FastAPI) (Completed)**

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
| **Phase 6** | **Risk Engine & Decision Framework** | 🟢 **Completed** | Milestone 6 |
| **Phase 7** | **Explainability & Reason Codes** | 🟢 **Completed** | Milestone 7 |
| **Phase 8** | **Fraud Detection API (FastAPI)** | 🟢 **Completed** | Milestone 8 (Current) |
| **Phase 9** | **Database & Persistence (PostgreSQL)** | ⚪ Pending | Next Milestone |
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

## ✅ Phase 7 Deliverable Checklist (Explainability & Reason Codes)

- [x] **Core Explainability Package (`ml/explainability/`)**:
  - Implemented typed, immutable data contracts in `ml/explainability/schemas.py` (`AttributionDirection`, `ReasonSource`, `ReasonSeverity`, `FeatureAttribution`, `ReasonCodeDetail`, `WaterfallStep`, `TransactionExplanation`).
  - Authored complete 55-feature explainability registry in `ml/explainability/config.py` defining display names, domain categorization, units, and plain-English risk/mitigating templates.
- [x] **Native TreeSHAP Engine (`ml/explainability/explainer.py`)**:
  - Leveraged frozen XGBoost compiled C++ TreeSHAP (`pred_contribs=True`) with zero external `shap` package dependencies, eliminating binary C-extension fragility and Windows OpenMP collisions.
  - Dynamically extracts baseline expected margin ($\phi_0 \approx 0.243697$) and verifies Lundberg Additivity ($\sum \phi_i + \phi_0 = \text{output\_margin}$) with $\max \text{diff} < 7.2 \times 10^{-6}$.
- [x] **Mathematically Complete Margin Waterfall**:
  - Constructs verified waterfall steps accounting for 100% of log-odds margin: Base Value $\to$ Selected Top Features $\to$ Residual ("Other feature contributions") $\to$ Final Margin display marker.
- [x] **Reason Code Synthesizer (`ml/explainability/reason_codes.py`)**:
  - Combines deterministic Phase 6 business rule triggers (`source="RULE"`) and top local TreeSHAP feature attributions (`source="MODEL"`).
  - Enforces strict precedence: Overriding Rules $\to$ Actionable Rules $\to$ Model Risk Drivers $\to$ Passive Monitoring Rules.
- [x] **Decision Provenance & Override Transparency**:
  - Distinctly exposes `model_score`, `baseline_action`, `rule_action`, `action`, and `is_overridden`.
  - Explicitly states when a review queue escalation was initiated by a business rule rather than implying model prediction.
- [x] **Raw-Value & Categorical Integrity**:
  - Preserves raw string categorical values (`merchant_category`, `job_category`) and unencoded numeric inputs; internal ordinal codes are never presented as business-meaningful values.
- [x] **Evaluator Integration (`RiskEvaluator`)**:
  - Added `explain_transaction()` and `evaluate_with_explanation()` preserving 100% backward compatibility for existing methods.
- [x] **Performance Benchmarking**:
  - Measured local single explanation latency ($\text{mean} = 25.18\text{ ms}$, $\text{median} = 24.29\text{ ms}$) and batch TreeSHAP throughput ($602.77\ \mu\text{s/row}$, $\approx 1,659\text{ explanations/sec}$).
- [x] **Comprehensive Testing & Artifact Invariance**:
  - Added 26 dedicated Phase 7 unit/integration tests; achieved 358 / 358 passing tests across the entire ML suite.
  - Verified SHA-256 artifact immutability across all frozen models, preprocessors, metadata, and Parquet partitions.
- [x] **Authoritative Documentation**:
  - Authored comprehensive Phase 7 report in `docs/explainability_report.md`.

---

## ✅ Phase 8 Deliverable Checklist (Fraud Detection API — FastAPI)

- [x] **FastAPI Application & Core Configuration**:
  - Implemented application entrypoint in `backend/app/main.py` with asynchronous lifespan manager for eager model loading and pre-warming.
  - Configured `backend/app/core/config.py` with strongly typed environment settings, artifact paths, API prefix (`/api/v1`), and CORS origins.
- [x] **Strict Pydantic v2 Schemas**:
  - Defined `TransactionPredictRequest` in `backend/app/schemas/predict.py` validating all 55 canonical and engineered feature columns with physical/domain bounds and optional metadata tolerance (`extra="allow"`).
  - Defined typed, standardized response models: `PredictionResponse`, `ReasonCodeResponse`, `RuleMatchResponse`, `FeatureAttributionResponse`, and `HealthResponse`.
- [x] **Service Layer & RiskEvaluator Reuse**:
  - Architected `RiskService` in `backend/app/services/risk_service.py` wrapping the frozen `RiskEvaluator` and standard 6-rule catalog (`get_standard_rule_catalog()`).
  - Integrated dependency injection hook `get_risk_service` for high-throughput, thread-safe inference without model reloading.
- [x] **Dual-Path REST Endpoints**:
  - Implemented `GET /health` and `GET /api/v1/health` delivering real-time service telemetry, champion model version provenance (`1.0.0`), and rule counts.
  - Implemented `POST /predict` and `POST /api/v1/predict` returning continuous score, 0–100 risk score, tri-tier decision (`APPROVE`/`REVIEW`/`BLOCK`), deterministic rule matches, and local TreeSHAP reason codes.
- [x] **Comprehensive Test Suite**:
  - Authored 21 dedicated API tests across `tests/unit/test_api_schemas.py`, `tests/unit/test_api_health.py`, and `tests/integration/test_api_predict.py`.
  - Achieved **379 / 379 passing tests** (358 ML tests + 21 API tests) with 100% pass rate.
  - Verified SHA-256 artifact immutability before and after API executions.
- [x] **Interactive Documentation & Runbooks**:
  - Updated `README.md` with local Uvicorn startup commands (`uvicorn backend.app.main:app --reload`), pytest commands, and curl examples.

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
- **Status**: Accepted (Phase 6).

### ADR-012: Native TreeSHAP Explanations & Margin Waterfall Transparency
- **Status**: Accepted (Phase 7).

### ADR-013: FastAPI Microservice Architecture & Zero-Mutation Inference
- **Context**: Upstream payment gateways and web dashboards require a low-overhead, strictly validated REST interface to evaluate transactions in real-time, retrieve explainable decision reason codes, and inspect rule telemetry without modifying frozen ML models or datasets.
- **Decision**: Build the REST API using **FastAPI** and **Pydantic v2**, managing `RiskEvaluator` and `RuleEngine` lifecycles via dependency injection. Pre-warm model artifacts in the lifespan context to minimize runtime cold starts. Expose both root-level (`/health`, `/predict`) and versioned (`/api/v1/health`, `/api/v1/predict`) endpoints. Validate the exact 55-feature schema while preserving transaction metadata passthrough.
- **Status**: Accepted (Phase 8).

---

## ⚠️ Known Constraints & Risk Register

1. **Analyst Review Scope**: Explanation payloads and reason codes provide interpretability and triage assistance for human investigators; they do not constitute statutory legal compliance certifications.
2. **Margin vs Probability Additivity**: TreeSHAP attributions are additive in raw log-odds margin space. Due to the non-linearity of the logistic sigmoid link function, individual feature contributions cannot be linearly summed in probability space.
3. **Review Queue Operational Sizing**: Hybrid rule overrides add manual review volume (+168.4% in OOT holdout). Real-world deployments must size analyst capacity accordingly.
4. **Static Cost Assumptions**: Current threshold benchmarks assume fixed unit costs ($C_{\text{FP}}=\$15, C_{\text{FN}}=\$200, C_{\text{REV}}=\$5$). Dynamic amount-weighted scoring is recommended for future financial optimization.

---

## ⏭ Next Step: Preparation for Phase 9 (Database & Persistence — PostgreSQL)

When approved to start Phase 9:
- Design and implement relational PostgreSQL schemas with SQLAlchemy ORM and Alembic migrations.
- Model `users`/`accounts`, `transactions`, `risk_evaluations`, `cases`, and immutable `audit_logs`.
- Persist incoming transactions and risk evaluation results asynchronously via the FastAPI service layer.
