# PROJECT_STATUS.md — Project Tracking & Status Dashboard

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
> **Last Updated:** Current Date (Phase 1 Implementation Completed)  
> **Current Active Phase:** **Phase 1 — Data Pipeline & Ingestion (Completed)**  

---

## 📊 Phase Progress Summary

| Phase | Description | Status | Target Completion |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Project Foundation & Architecture** | 🟢 **Completed** | Milestone 0 |
| **Phase 1** | **Data Pipeline & Ingestion** | 🟢 **Completed** | Milestone 1 (Current) |
| **Phase 2** | **Exploratory Data Analysis (EDA) & Insights** | ⚪ Pending | Next Milestone |
| **Phase 3** | **Behavioral Feature Engineering** | ⚪ Pending | Phase 3 |
| **Phase 4** | **Baseline & Advanced ML Models** | ⚪ Pending | Phase 4 |
| **Phase 5** | **Imbalance Handling & Cost Optimization** | ⚪ Pending | Phase 5 |
| **Phase 6** | **Risk Engine & Decision Framework** | ⚪ Pending | Phase 6 |
| **Phase 7** | **Explainability & Reason Codes** | ⚪ Pending | Phase 7 |
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

## ✅ Phase 1 Deliverable Checklist

- [x] Documented dataset provenance: Synthetic Sparkov Credit Card benchmark dataset.
- [x] Established automated acquisition workflow via `ml/data/download.py` saving raw CSVs to `data/raw/benchmark/`.
- [x] Defined canonical 15-field transaction schema and validation engine in `ml/data/schema.py`.
- [x] Mapped source `cc_num` identifier to `account_id` without assuming cryptographic hashing.
- [x] Preserved source timestamp semantics without forcing unverified UTC assumptions.
- [x] Built end-to-end ingestion and time-aware splitting pipeline in `ml/data/pipeline.py`.
- [x] Applied deterministic multi-key chronological sorting (`unix_time`, `transaction_id`).
- [x] Executed Out-of-Time (OOT) temporal split: 70% Train (1,296,675 rows), 15% Validation (277,859 rows), 15% Test (277,860 rows).
- [x] Validated zero lookahead leakage ($\max(t_{\text{train}}) \le \min(t_{\text{val}}) \le \min(t_{\text{test}})$).
- [x] Serialized clean partitioned datasets in Apache Parquet format under `data/processed/benchmark/`.
- [x] Generated comprehensive summary metrics report in `docs/data_pipeline_report.md`.
- [x] Built and verified 7 automated unit tests in `tests/ml/test_data_pipeline.py`.

---

## 🏛 Architectural Decision Records (ADRs)

### ADR-001: Modular Monorepo Scaffolding
- **Decision**: Adopt a modular directory structure under a single repository, keeping concerns strictly decoupled into `backend/`, `ml/`, `frontend/`, `database/`, and `tests/`.
- **Status**: Accepted.

### ADR-002: Lightweight Phase-by-Phase Dependency Management
- **Decision**: Defer dependency installations and full manifests until the specific phase where those libraries are required.
- **Status**: Accepted.

### ADR-003: Measurable Non-Functional Requirements (Latency & Throughput)
- **Decision**: Treat latency and throughput as measurable performance goals to be benchmarked and tuned during Phase 10 (Real-Time Detection & Benchmarking).
- **Status**: Accepted.

### ADR-004: Calibrated 0–100 Risk Scoring with Tri-Tier Decision Policy
- **Decision**: Map calibrated model outputs to an intuitive 0–100 integer score with three discrete operational actions: `APPROVE` (<35), `REVIEW` (35–74), and `BLOCK` (≥75).
- **Status**: Accepted.

### ADR-005: Time-Aware Out-of-Time (OOT) Splitting & Zero-Leakage Protocol
- **Context**: Random K-fold splitting in financial transaction streams causes severe temporal lookahead bias, inflating offline metrics.
- **Decision**: Enforce strict chronological ordering (multi-key sorted by `unix_time` and `transaction_id`) and Out-of-Time partitioning (70% Train, 15% Val, 15% Test). No future transaction is ever present during training or validation fits.
- **Status**: Accepted (Phase 1).

### ADR-006: Canonical 15-Field Transaction Schema & Columnar Parquet Storage
- **Context**: Ingesting varying raw formats requires a single, type-enforced internal contract for downstream feature engineering and REST APIs.
- **Decision**: Standardize on a 15-field canonical schema stored in Apache Parquet format for fast columnar I/O, strict type enforcement, and zero disk bloat.
- **Status**: Accepted (Phase 1).

---

## ⚠️ Known Constraints & Risk Register

1. **Extreme Class Imbalance**: The dataset exhibits a 0.521% overall fraud rate (0.579% in Train, 0.439% in Val, 0.333% in Test). All modeling evaluations in Phase 4 & 5 must prioritize PR-AUC, Recall at low FPR, and cost-sensitive utility metrics.
2. **Synthetic Domain Boundaries**: Sparkov is a synthetic credit card transaction benchmark. In Phase 10 & 11, we will complement this with targeted synthetic real-time attack generators for live simulation.

---

## ⏭ Next Step: Preparation for Phase 2 (Exploratory Data Analysis & Domain Insights)

When approved to start Phase 2:
- Conduct deep statistical profiling across temporal, merchant, categorical, geographical, and monetary distributions.
- Analyze fraud concentration patterns (e.g., high-risk MCC categories, nighttime velocity spikes, transaction amount distributions).
- Author a comprehensive, insight-rich Exploratory Data Analysis report with publication-ready charts and domain conclusions.
