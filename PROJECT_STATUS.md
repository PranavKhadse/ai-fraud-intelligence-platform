# PROJECT_STATUS.md — Project Tracking & Status Dashboard

> **Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
> **Last Updated:** Current Date (Phase 0 Implementation)  
> **Current Active Phase:** **Phase 0 — Project Foundation and Architecture**  

---

## 📊 Phase Progress Summary

| Phase | Description | Status | Target Completion |
| :--- | :--- | :--- | :--- |
| **Phase 0** | **Project Foundation & Architecture** | 🟡 **In Progress** | Current Milestone |
| **Phase 1** | **Data Pipeline & Ingestion** | ⚪ Pending | Next Milestone |
| **Phase 2** | **Exploratory Data Analysis (EDA) & Insights** | ⚪ Pending | Phase 2 |
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

## ✅ Phase 0 Deliverable Checklist

- [x] Inspect workspace tooling (Python 3.11.7, Node v24.19.0, npm 11.17.0, Git repository status).
- [x] Configure `.gitignore` with full coverage for Python, Node, data files, models, logs, IDEs, and environments.
- [x] Create `.env.example` template with clean variable placeholders and zero hardcoded secrets.
- [x] Author comprehensive `README.md` covering problem statement, capabilities, tech stack, and high-level architecture.
- [x] Author `PROJECT_SPEC.md` as the authoritative single source of truth for all requirements and roadmap phases.
- [x] Author `ARCHITECTURE.md` detailing system topology, component responsibilities, and sequence data flows.
- [x] Author `PROJECT_STATUS.md` tracking active progress, ADRs, and upcoming milestones.
- [x] Create modular directory scaffolding with `.gitkeep` files.
- [ ] Perform Phase 0 verification (tree check, zero secrets scan, gitignore test).
- [ ] Stage and commit Phase 0 cleanly in Git (`feat(phase-0): project foundation and architecture`).

---

## 🏛 Architectural Decision Records (ADRs)

### ADR-001: Modular Monorepo Scaffolding
- **Context**: Need a scalable structure supporting backend API, machine learning pipelines, database migrations, frontend UI, and infrastructure.
- **Decision**: Adopt a modular directory structure under a single repository, keeping concerns strictly decoupled into `backend/`, `ml/`, `frontend/`, `database/`, and `tests/`.
- **Status**: Accepted.

### ADR-002: Lightweight Phase-by-Phase Dependency Management
- **Context**: Installing heavy dependencies prematurely leads to version conflicts and bloated environments.
- **Decision**: Defer dependency installations and full manifests until the specific phase where those libraries are required (e.g., ML libraries in ML phases, FastAPI in API phases).
- **Status**: Accepted.

### ADR-003: Measurable Non-Functional Requirements (Latency & Throughput)
- **Context**: Early commitment to rigid sub-millisecond targets can cause premature architectural optimization.
- **Decision**: Treat latency and throughput as measurable performance goals to be benchmarked and tuned during Phase 10 (Real-Time Detection & Benchmarking).
- **Status**: Accepted.

### ADR-004: Calibrated 0–100 Risk Scoring with Tri-Tier Decision Policy
- **Context**: Raw ML probabilities are difficult for operational teams to interpret directly without clear business policy mappings.
- **Decision**: Map calibrated model outputs to an intuitive 0–100 integer score with three discrete operational actions: `APPROVE` (<35), `REVIEW` (35–74), and `BLOCK` (≥75).
- **Status**: Accepted.

---

## ⚠️ Known Constraints & Risk Register

1. **Severe Class Imbalance**: Fraud instances will represent a tiny fraction ($<1\%$) of total transactions. Evaluation metrics must strictly prioritize PR-AUC and cost-weighted loss matrices over standard accuracy.
2. **Data Leakage Risk in Feature Engineering**: Time-series features (velocity, historical deviation) must be calculated strictly with temporal boundaries to prevent future data leakage into training sets.

---

## ⏭ Next Step: Preparation for Phase 1 (Data Pipeline & Ingestion)

When Phase 0 is validated and approved:
- Plan the dataset strategy (structured synthetic transaction generation and schema validation).
- Define transaction data schemas (temporal fields, merchant categories, amounts, cardholder profiles, geo-coordinates).
- Establish partitioning and data versioning principles.
