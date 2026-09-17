# PROJECT_STATUS.md — Project Tracking & Status Dashboard

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform
> **Last Updated:** Current Date (Phase 11 Implementation Completed)
> **Current Active Phase:** **Phase 11 — Fraud Intelligence Dashboard & What-If Simulator (Completed)**

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
| **Phase 8** | **Fraud Detection API (FastAPI)** | 🟢 **Completed** | Milestone 8 |
| **Phase 9** | **Database & Persistence (PostgreSQL)** | 🟢 **Completed** | Milestone 9 |
| **Phase 10** | **Real-Time Detection & Benchmarking** | 🟢 **Completed** | Milestone 10 |
| **Phase 11** | **Fraud Intelligence Dashboard** | 🟢 **Completed** | Milestone 11 |
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

## ✅ Phase 9 Deliverable Checklist (Database & Persistence — PostgreSQL)

- [x] **Database Engine, Pool & Session Management (`backend/app/db/`)**:
  - Implemented lazy `AsyncEngine` with QueuePool connection pooling in `backend/app/db/session.py`.
  - Built `get_db_session` FastAPI dependency guaranteeing request isolation, automatic rollback on unhandled exceptions, and explicit commit semantics.
  - Implemented graceful pool disposal in FastAPI lifespan shutdown hook.
  - Added sanitized `check_db_health()` connectivity probe (`SELECT 1`) in `backend/app/db/health.py`.
- [x] **SQLAlchemy 2.0 ORM Relational Models (`backend/app/db/models/`)**:
  - Modeled 6 core tables: `transactions`, `risk_evaluations`, `evaluation_rule_matches`, `evaluation_reason_codes`, `evaluation_feature_attributions`, and `audit_logs`.
  - Declared `UUIDPrimaryKeyMixin` and `TimestampMixin` for base modeling.
  - Enforced strict relational foreign keys: `ON DELETE RESTRICT` from evaluations to transactions; `ON DELETE CASCADE` from explanations to evaluations.
  - Enforced database check constraints: `chk_transactions_amount_positive`, `chk_risk_evaluations_model_score`, and `chk_risk_evaluations_risk_score`.
  - Declared partial unique index `uq_transactions_external_tx_id` on `transactions(external_transaction_id) WHERE external_transaction_id IS NOT NULL`.
- [x] **Alembic Migration Foundation (`backend/alembic/`)**:
  - Configured `alembic.ini`, `backend/alembic/env.py`, and `script.py.mako`.
  - Created baseline revision `0001_initial_core_tables.py` creating all 6 core tables, constraints, and indexes.
  - Created revision `0002_add_unique_index_external_tx_id.py` managing the partial unique index.
  - Verified offline SQL DDL generation without requiring a live database connection.
- [x] **Repository Layer (`backend/app/repositories/`)**:
  - Implemented typed asynchronous repositories: `TransactionRepository`, `RiskEvaluationRepository`, `RuleMatchRepository`, `ReasonCodeRepository`, `FeatureAttributionRepository`, `AuditLogRepository`.
  - Added `get_with_evaluations_by_external_id()` using `selectinload` for eager fetching of child explanation collections.
  - Designed domain exception hierarchy: `PersistenceError`, `PersistenceConflictError`, `PersistenceNotFoundError`.
- [x] **Unit of Work & Orchestration Service (`backend/app/services/`)**:
  - Implemented `FraudPersistenceUnitOfWork` coordinating all 6 repositories under a single transaction boundary with async context manager safety.
  - Built `FraudPersistenceService` orchestrating strict staging order (`Transaction` $\to$ `flush()` $\to$ `RiskEvaluation` $\to$ `flush()` $\to$ Child collections & `AuditLog` $\to$ `commit()`).
  - Added selective `IntegrityError` discriminator translating unique index conflicts to `PersistenceConflictError` while isolating unrelated database violations into `PersistenceError`.
- [x] **API Persistence Mapping & Idempotency Protection (`backend/app/api/`, `backend/app/services/`)**:
  - Implemented `RiskPersistenceMapper` for zero-database in-memory mapping from request/response to typed commands.
  - Built deterministic canonical SHA-256 request fingerprinting (`compute_request_fingerprint`) and payload equivalence verification (`is_payload_equivalent`).
  - Built complete `PredictionResponse` reconstruction from database entities (`reconstruct_prediction_response`).
  - Wired `/predict` and `/api/v1/predict` with pre-inference replay fast path (200 OK), payload conflict detection (409 Conflict), correlation context extraction, and concurrency race recovery.
  - Implemented 4-way timestamp semantics and strict 128-character ID length boundaries.
- [x] **Comprehensive Testing & Validation**:
  - 231 unit tests passing across ORM models, session, health, repositories, Unit of Work, mapper, endpoint hardening, and idempotency.
  - 35 integration tests passing (including 24 database persistence integration tests against live PostgreSQL).
  - 624 / 624 total tests passing across full repository test suite.
  - Documented authoritative Phase 9 report in `docs/phase_9_persistence_completion_report.md`.

---

## ✅ Phase 10 Deliverable Checklist (Real-Time Detection & Benchmarking)

- [x] **Increment 10.1: Core Latency Metrics & Component Profiling Infrastructure**:
  - Implemented typed `BenchmarkConfig` and `TargetSLA` reference models with boundaries and positive-value validations in `backend/app/benchmarking/config.py`.
  - Implemented mathematical linear-interpolation percentile engine (`calculate_percentiles`), comprehensive `LatencyMetrics`, `ThroughputMetrics`, `ComponentLatencyBreakdown`, and `SLACompliance` in `backend/app/benchmarking/metrics.py`.
  - Built high-resolution nanosecond async `MetricCollector` in `backend/app/benchmarking/collector.py`.
  - Built 7-stage `PipelineProfiler` in `backend/app/benchmarking/profiler.py` decomposing request validation, feature prep, ML scoring, TreeSHAP explainability, rule engine, mapper, and PostgreSQL persistence.
  - Added 61 unit tests covering percentiles, distributions, and stage isolation.
- [x] **Increment 10.2: Asynchronous Multi-Concurrency Load Generator & Replay Harness**:
  - Implemented in-memory dataset pre-loading and caching in `BenchmarkRunner` (`backend/app/benchmarking/runner.py`) eliminating disk I/O distortion from latency measurements.
  - Implemented deterministic collision-free 128-character unique `external_transaction_id` generator.
  - Enforced bounded concurrency via `asyncio.Semaphore` with dual in-process ASGI and network HTTP transport modes.
  - Implemented dedicated two-pass idempotency replay benchmark evaluating cache hit latency vs fresh ML scoring.
  - Added 15 tests (8 unit, 7 integration) verifying ASGI load generation, rate limiting, and replay verification.
- [x] **Increment 10.3: Cold-Start / Warm-Up Isolation, Concurrency Sweep & Persistence Ablation**:
  - Built `BenchmarkSuite` (`backend/app/benchmarking/suite.py`) coordinating cold-start isolation, 50-request unmeasured warm-up priming, and multi-concurrency matrix sweeps across $C \in \{1, 2, 4, 8, 16\}$.
  - Implemented multi-tier persistence ablation measuring Mode A (Full API + PostgreSQL), Mode B (In-Memory Pipeline), and Mode C (Idempotent DB Replay), quantifying persistence overhead and replay speedups.
  - Implemented 7-stage component bottleneck diagnostics ranking primary/secondary bottlenecks.
  - Added 13 tests (8 unit, 5 integration) verifying sweep scaling ratios, ablation formulas, and full suite execution.
- [x] **Increment 10.4: SLA Compliance, Benchmark CLI, Live Execution & Documentation**:
  - Built `BenchmarkReporter` in `backend/app/benchmarking/reporter.py` generating console ASCII tables, machine-readable JSON reports (`docs/benchmark_results.json`), CSV exports (`docs/benchmark_results.csv`), and GFM tables.
  - Built standalone reproducible CLI tool in `scripts/benchmark.py` supporting custom requests, concurrency sequences, warmups, rates, transports, seeds, and output targets.
  - Executed live empirical benchmark on active system stack (16-core AMD64 Windows, Python 3.11, PostgreSQL), recording 100% genuine measurements (0% error rate across 1,000 transactions, peak throughput 26.5 TPS at $C=4$, 21.8 ms persistence overhead, 2.84x replay speedup, and TreeSHAP primary in-memory bottleneck at 41.4%).
  - Authored comprehensive 17-section documentation in `docs/phase_10_realtime_detection_benchmarking.md` and `docs/phase_10_completion_report.md`.
  - Expanded test suite to **727 total passing tests** (103 Phase 10 tests) with 100% pass rate.

---

## ✅ Phase 11 Deliverable Checklist (Fraud Intelligence Dashboard & What-If Simulator)

- [x] **Increment 11.1: Dashboard Foundation & UI Scaffolding**:
  - Initialized modular React 18 + TypeScript + Vite frontend under `frontend/` with structured layout, clean routing, and production build tooling.
  - Implemented sleek dark-mode design system with curated CSS custom properties (`#0B0F19` canvas, `#111827` cards, cyan/emerald/amber/rose risk badges, smooth animations).
  - Built backend high-level overview endpoint `GET /api/v1/dashboard/overview` providing 24h summary metrics (total count, approval/review/block rates, average risk score, fraud loss avoided).
- [x] **Increment 11.2: Near-Real-Time Risk Intelligence & Live Transaction Feed**:
  - Implemented `GET /api/v1/dashboard/feed` supporting pagination, search, risk tier filtering (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), sorting, and time window parameters.
  - Created interactive `LiveTransactionFeed` with auto-polling toggle (5s / 10s / 30s / off) and visual near-real-time indicator.
- [x] **Increment 11.3: Transaction Investigation & Deep Inspection**:
  - Implemented `GET /api/v1/dashboard/transactions/{tx_id}` in `DashboardRepository` delivering transaction metadata, persisted risk evaluations, 55-feature snapshots, TreeSHAP attributions, rule matches, plain-English reason codes, and audit trails.
  - Built slide-over `TransactionDrawer` with formatted currency, location travel speeds, risk score meters, feature category cards, and complete audit history.
- [x] **Increment 11.4: Explainability Visualizer & Trend Analytics**:
  - Implemented `GET /api/v1/dashboard/analytics` delivering 10-bucket risk score histograms ($0\text{--}9, \dots, 90\text{--}100$ with boundary 100 handling), hourly/daily volume trends, and top-triggered rules.
  - Built interactive `AnalyticsView` and `ShapWaterfall` chart faithfully visualizing base log-odds margin ($\phi_0 \approx 0.2437$), individual feature attributions, residual delta, and final model score.
- [x] **Increment 11.5: What-If Transaction Simulator, Hardening, Testing & Documentation**:
  - Created `POST /api/v1/dashboard/simulate` executing entirely in-memory using production `RiskEvaluator` and `RuleEngine` with zero persistence calls (`FraudPersistenceService` isolated).
  - Built 55-feature counterfactual editor grouped across 7 domain categories with live modified badges, category resets, and preset scenarios.
  - Implemented side-by-side comparison summary card ($\Delta\text{Risk Score}$, $\Delta\text{Model Score}$, tier changes, action changes) and deduplicated rule impact analysis (`NEWLY_TRIGGERED`, `RESOLVED`, `PERSISTENT`, `NEITHER`).
  - Added seamless "Simulate in What-If" action from `TransactionDrawer` pre-populating baseline transaction snapshot.
  - Added 23 dedicated Phase 11.5 backend tests (schema validations, NaN/Infinity rejections, boundary checks, zero-write invariant tests, baseline immutability); achieved **819 / 819 passing tests** with 0 regressions.
  - Authored comprehensive documentation in `docs/phase_11_fraud_intelligence_dashboard_report.md`.

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
- **Status**: Accepted (Phase 8).

### ADR-014: Relational PostgreSQL Persistence, Unit of Work & Idempotency Architecture
- **Status**: Accepted (Phase 9).

### ADR-015: Empirical High-Resolution Latency Profiling & Multi-Tier Benchmark Architecture
- **Status**: Accepted (Phase 10).

### ADR-016: Read-Only What-If Simulation Architecture & Zero-Write State Governance
- **Context**: Fraud analysts require counterfactual simulation to evaluate how hypothetical feature alterations (e.g. higher velocity, altered location, abnormal amount) impact ML model scores, TreeSHAP attributions, and rule triggers without contaminating the production database or creating audit noise.
- **Decision**: Design `POST /api/v1/dashboard/simulate` as an entirely in-memory evaluation pipeline reusing the singleton `RiskService`, `RiskEvaluator`, and `RuleEngine`. Enforce a strict zero-write guarantee: baseline queries are read-only (`DashboardRepository.get_transaction_detail`), `FraudPersistenceService` is never invoked, no `INSERT`/`UPDATE`/`DELETE` queries or audit logs are executed, and baseline entity states are verified immutable.
- **Status**: Accepted (Phase 11).

---

## ⚠️ Known Constraints & Risk Register

1. **Analyst Review Scope**: Explanation payloads and reason codes provide interpretability and triage assistance for human investigators; they do not constitute statutory legal compliance certifications.
2. **Margin vs Probability Additivity**: TreeSHAP attributions are additive in raw log-odds margin space. Due to the non-linearity of the logistic sigmoid link function, individual feature contributions cannot be linearly summed in probability space.
3. **Review Queue Operational Sizing**: Hybrid rule overrides add manual review volume (+168.4% in OOT holdout). Real-world deployments must size analyst capacity accordingly.
4. **Static Cost Assumptions**: Current threshold benchmarks assume fixed unit costs ($C_{\text{FP}}=\$15, C_{\text{FN}}=\$200, C_{\text{REV}}=\$5$). Dynamic amount-weighted scoring is recommended for future financial optimization.
5. **Distributed Ambiguous Commit Outcome**: If network connectivity drops while awaiting PostgreSQL `COMMIT` acknowledgement, the outcome is inherently ambiguous across distributed nodes. The idempotency design safely handles both outcomes upon subsequent retry: if the transaction committed, the retry replays the result (`200 OK`); if the commit was rolled back by PostgreSQL, the retry performs clean evaluation and persistence.
6. **Local Single-Node CPU Contention**: Synchronous TreeSHAP execution and PostgreSQL commits on a shared local host constrain peak throughput to ~26.5 TPS. Offloading persistence and TreeSHAP attributions to background asynchronous queues is recommended for high-volume (>1,000 TPS) deployments.
7. **Near-Real-Time Feed Polling**: The Live Transaction Feed operates via configurable near-real-time client-side polling (5s/10s/30s) rather than WebSocket streaming. For high-volume (>5,000 TPS) streams, WebSocket or Server-Sent Events (SSE) should be evaluated.

---

## ⏭ Next Step: Preparation for Phase 12 (Human Review & Case Management)

When approved to start Phase 12:
- Architect human review queue and analyst workflow management.
- Implement case assignment, manual dispositioning (`CONFIRMED_FRAUD`, `FALSE_POSITIVE`, `DISMISSED`), and disposition history.
- Capture analyst feedback for downstream active learning and retraining loops.

