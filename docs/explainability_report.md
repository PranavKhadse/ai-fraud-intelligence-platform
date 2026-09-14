# Phase 7: Explainability, Reason Codes & Decision Transparency Report

**Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform  
**Phase:** Phase 7 (Explainability & Reason Codes)  
**Champion Model:** Frozen XGBoost Champion (`ml/models/artifacts/champion_model.joblib`)  
**Status:** Implemented, Tested & Verified Across Test Suites  

---

## 1. Executive Summary & Objective

Phase 7 implements **Explainability, Reason Codes & Decision Transparency**, introducing local feature attributions, mathematically complete margin waterfalls, and standardized human-readable reason codes to the risk intelligence platform.

The objective of Phase 7 is to provide fraud analysts and investigators with interpretable diagnostic records that support analyst review, troubleshooting, and internal auditability for evaluated transactions. By combining native gradient-boosted tree feature attributions with deterministic business rule outcomes, the platform explains *why* a transaction was flagged, routed, or cleared—clearly distinguishing continuous statistical model drivers from deterministic policy overrides.

> [!NOTE]
> This module is designed strictly as an analyst-support and diagnostic transparency tool to assist human fraud investigators and system auditability. It does not constitute a statutory compliance certification, legal warranty, or autonomous regulatory disposition.

---

## 2. Implemented Architecture & Components

The explainability framework is organized under [`ml/explainability/`](../ml/explainability/) and integrated into [`RiskEvaluator`](../ml/risk_engine/evaluator.py):

```
                               ┌─────────────────────────────────────────┐
                               │       Incoming Transaction Payload      │
                               └────────────────────┬────────────────────┘
                                                    │
                         ┌──────────────────────────┴──────────────────────────┐
                         ▼                                                     ▼
           ┌───────────────────────────┐                         ┌───────────────────────────┐
           │      Risk Evaluator       │                         │    TreeSHAP Explainer     │
           │  (XGBoost Model + Rules)  │                         │ (Native XGBoost C++ SHAP) │
           └─────────────┬─────────────┘                         └─────────────┬─────────────┘
                         │                                                     │
                         ▼                                                     ▼
           ┌───────────────────────────┐                         ┌───────────────────────────┐
           │      Decision Result      │                         │ Raw Attributions (ϕ₁..ϕ₅₅)│
           │   (APPROVE/REVIEW/BLOCK   │                         │  + Base Value (ϕ₀ = 0.24) │
           │   + Triggered Rules)      │                         └─────────────┬─────────────┘
           └─────────────┬─────────────┘                                       │
                         │                                                     │
                         └──────────────────────────┬──────────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │         Reason Code Synthesizer         │
                               │  - Combines Rule Matches (Precedence 1) │
                               │  - Top-K Risk Factors (ϕ > 0)           │
                               │  - Top-M Mitigating Factors (ϕ < 0)     │
                               │  - Plain-English Template Interpolation │
                               │  - Step-by-Step Waterfall Construction  │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                               ┌─────────────────────────────────────────┐
                               │       Transaction Explanation API       │
                               │   (Immutable, JSON-Serializable, Typed) │
                               └─────────────────────────────────────────┘
```

### 2.1 Core Modules & Responsibilities

1. [`ml/explainability/schemas.py`](../ml/explainability/schemas.py):
   * Defines strongly-typed, immutable dataclasses for explainability data structures.
   * Enums: [`AttributionDirection`](../ml/explainability/schemas.py) (`RISK_INCREASING`, `MITIGATING`), [`ReasonSource`](../ml/explainability/schemas.py) (`MODEL`, `RULE`), and [`ReasonSeverity`](../ml/explainability/schemas.py) (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`).
   * [`FeatureAttribution`](../ml/explainability/schemas.py): Immutable record of feature name, display name, unencoded raw value, margin-space SHAP contribution ($\phi_i$), directional flag, relative percentage contribution, and rank.
   * [`ReasonCodeDetail`](../ml/explainability/schemas.py): Standardized reason code with code string, headline, plain-English description, domain category, source origin (`RULE` vs `MODEL`), severity, and rank.
   * [`WaterfallStep`](../ml/explainability/schemas.py): Verified additive step in margin log-odds space.
   * [`TransactionExplanation`](../ml/explainability/schemas.py): Comprehensive explanation payload containing probability score, output margin, base margin, calibrated risk score, risk tier, baseline action, final action, top risk factors, top mitigating factors, reason codes, waterfall steps, override provenance, and triggered rule matches.

2. [`ml/explainability/config.py`](../ml/explainability/config.py):
   * Canonical feature registry mapping all 55 predictive columns to display names, domain categories (`AMOUNT`, `VELOCITY`, `GEOGRAPHY`, `ACCOUNT_HISTORY`, `TEMPORAL`, `CATEGORICAL`), measurement units, and plain-English risk/mitigating explanation templates.

3. [`ml/explainability/explainer.py`](../ml/explainability/explainer.py):
   * [`TreeSHAPExplainer`](../ml/explainability/explainer.py): Wraps the frozen XGBoost champion model's underlying booster to perform native TreeSHAP attribution extraction without external third-party library dependencies.

4. [`ml/explainability/reason_codes.py`](../ml/explainability/reason_codes.py):
   * [`ReasonCodeGenerator`](../ml/explainability/reason_codes.py): Synthesizes rule matches and top TreeSHAP factors into prioritized, human-readable reason codes, enforcing source separation, deduplication, and override transparency.

5. [`ml/risk_engine/evaluator.py`](../ml/risk_engine/evaluator.py):
   * Extended [`RiskEvaluator`](../ml/risk_engine/evaluator.py) with [`explain_transaction()`](../ml/risk_engine/evaluator.py) and [`evaluate_with_explanation()`](../ml/risk_engine/evaluator.py), maintaining 100% backward compatibility for existing evaluation pipelines.

---

## 3. Decision Semantics & Rule Precedence Transparency

A core requirement of Phase 7 is the unambiguous representation of decision provenance. When a transaction is escalated to manual review by a business rule, the system explicitly documents that the escalation was initiated by a deterministic policy rule rather than misrepresenting the outcome as an ML model prediction.

### 3.1 Five-Component Decision Decomposition

Every explanation payload exposes five distinct decision parameters:

| Parameter | Type | Semantic Definition |
| :--- | :--- | :--- |
| `model_score` | `float` | Continuous ML fraud probability $p(x) \in [0.0, 1.0]$. |
| `baseline_action` | `DecisionAction` | Action dictated solely by the ML model score against policy thresholds (`APPROVE`, `REVIEW`, `BLOCK`). |
| `rule_action` | `Optional[RuleOutcome]` | Action outcome of the highest-precedence matched deterministic rule (`BLOCK`, `REVIEW`, `MONITOR`, or `None`). |
| `action` | `DecisionAction` | Final resolved operational disposition after applying deterministic precedence hierarchy. |
| `is_overridden` | `bool` | Boolean flag confirming whether a business rule overrode the model's baseline disposition. |

### 3.2 Decision Interaction Scenarios

1. **Rule Escalation Override (Model APPROVE + Rule REVIEW):**
   * *Example:* Transaction with low model score ($p = 0.015 < 0.35$) matching `RULE_VELOCITY_BURST_REVIEW`.
   * `model_score = 0.0152`, `baseline_action = APPROVE`, `rule_action = REVIEW`, `action = REVIEW`, `is_overridden = True`.
   * *Reason Code:* `"Triggered business rule: 1-hour transaction count (5) exceeds the review threshold of 4 transactions. [Decision escalated to REVIEW by policy rule]."` (`source="RULE"`).
   * *Transparency Principle:* The model's low risk score is faithfully recorded, while the review queue routing is attributed to the deterministic velocity policy.

2. **Protected Model Block (Model BLOCK + Rule REVIEW):**
   * *Example:* Transaction with high model score ($p = 0.912 \ge 0.78$) matching `RULE_VELOCITY_BURST_REVIEW`.
   * `model_score = 0.9124`, `baseline_action = BLOCK`, `rule_action = None`, `action = BLOCK`, `is_overridden = False`.
   * *Transparency Principle:* Lower-tier review rules never downgrade an automated model block. The rule match is recorded in `rules_triggered` for audit logging without overriding the hard block.

3. **Passive Monitoring (Model APPROVE + Rule MONITOR):**
   * *Example:* Transaction matching `RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR`.
   * `model_score = 0.0240`, `baseline_action = APPROVE`, `rule_action = None`, `action = APPROVE`, `is_overridden = False`.
   * *Transparency Principle:* Monitoring rules provide passive telemetry and audit annotations (`severity="INFO"`) without modifying the transaction disposition.

---

## 4. TreeSHAP Mathematical Semantics & Output Contract

### 4.1 Native XGBoost Inference
Explainability is powered by the frozen XGBoost booster's compiled C++ TreeSHAP engine:
```python
dmat = xgb.DMatrix(X_trans, feature_names=PREDICTIVE_FEATURE_COLUMNS)
contribs = booster.predict(dmat, pred_contribs=True)
```

* **Input:** Dense preprocessed float array of shape $(N, 55)$ generated by `TreePreprocessor.transform(X)`.
* **Output:** Array of shape $(N, 56)$ where:
  * Columns `0` through `54`: Local feature attributions ($\phi_1, \dots, \phi_{55}$) aligned 1-to-1 with [`PREDICTIVE_FEATURE_COLUMNS`](../ml/models/config.py).
  * Column `55` (56th column): Global TreeSHAP base value $\phi_0$.

### 4.2 Margin Space vs Probability Space

> [!IMPORTANT]
> TreeSHAP feature attributions operate strictly in **raw log-odds margin space** ($\mathbb{R}$), not probability space ($[0, 1]$). SHAP values must not be described as "percentage probability additions."

The mathematical relationship between the explanation components is defined as:

1. **Margin-Space Additivity:**
   $$\text{output\_margin}(x) = \phi_0 + \sum_{i=1}^{55} \phi_i(x)$$

2. **Probability Reconstruction (Sigmoid Link):**
   $$\text{model\_score}(x) = \sigma(\text{output\_margin}(x)) = \frac{1}{1 + e^{-\text{output\_margin}(x)}}$$

3. **Dynamic Base Value Extraction:**
   * The base value $\phi_0 = E[f(X)]$ represents the expected log-odds margin over the training background distribution.
   * It is dynamically extracted from `contribs[0, 55]` and is never hardcoded.
   * *Observed benchmark value on the champion model:* $\phi_0 \approx 0.24369707703590393$ ($\sigma(\phi_0) \approx 0.5606$, reflecting the `scale_pos_weight = 171.75` class weighting).

---

## 5. Additivity & Mathematically Complete Margin Waterfall

To provide fraud analysts with an interpretable step-by-step diagnostic view, the platform constructs a mathematically complete waterfall that accounts for 100% of the model's output margin.

```
Expected Base Margin (ϕ₀ = +0.2437)
  │
  ├─ (+) Transaction Amount: +3.3885
  ├─ (-) 24-Hour Max Amount: -1.2890
  ├─ (+) Amount to Historical Mean Ratio: +0.8917
  ├─ (-) Merchant Category: -0.7818
  ├─ (+) 1-Hour Velocity: +0.1032
  │
  ├─ Residual ("Other feature contributions"): -6.0999
  │
  ▼
Final Output Margin (-4.1695) ──[Sigmoid]──► Fraud Probability: 0.0152 (Risk Score: 2/100)
```

### 5.1 Waterfall Step Structure

1. **Base Step (`step_type="base"`):**  
   Initial expected margin $\phi_0$ across the training population.
2. **Selected Top Feature Steps (`step_type="feature"`):**  
   Top positive risk drivers ($\phi_i > 0$) and top mitigating factors ($\phi_i < 0$) sorted by absolute magnitude $|\phi_i|$.
3. **Residual Step (`step_type="residual"`):**  
   `"Other feature contributions"`, capturing the exact sum of all unselected feature attributions:
   $$\text{residual} = \sum_{j \notin \text{selected}} \phi_j$$
4. **Final Display Marker (`step_type="final"`):**  
   `"Final Output Margin"`, with `contribution = 0.0` and `cumulative_margin = output_margin`.
   * *Mathematical Invariant:* The final marker step is a terminal display element with contribution 0.0 and is excluded from mathematical additive summation:
     $$\text{base\_value} + \sum_{f \in \text{selected}} \phi_f + \text{residual} = \text{output\_margin}$$

### 5.2 Observed Reconstruction Tolerances

Empirically verified across 10,000 transactions from the validation and out-of-time test partitions:
* **Margin Additivity Max Discrepancy:** $\max \left| \sum \phi_i + \phi_0 - \text{output\_margin} \right| \approx 7.195 \times 10^{-6}$ (floating-point IEEE-754 precision).
* **Mean Margin Discrepancy:** $\approx 1.390 \times 10^{-6}$.
* **Probability Reconstruction Discrepancy:** $\max \left| p(x) - \sigma(\text{output\_margin}) \right| < 1.0 \times 10^{-8}$.

---

## 6. Raw-Value Integrity & Categorical Handling

The explainability layer enforces strict separation between internal model encoding and human-readable feature presentation:

1. **Categorical Representation:**
   * The champion model preprocessor encodes categorical features (`merchant_category`, `job_category`) using ordinal integers (`0.0`, `1.0`, etc.) fitted on training data.
   * [`TreeSHAPExplainer`](../ml/explainability/explainer.py) preserves the raw, unencoded string values from the transaction input.
   * Explanations and reason codes format the actual business category (e.g. `'shopping_net'`, `'grocery_pos'`).
   * **Rule:** Internal ordinal integer codes are **never** presented as business-meaningful values.
2. **Robust Edge-Case Support:**
   * **Missing / Null Categoricals:** Handled safely via null-safe fallbacks without raising exceptions.
   * **Unseen / OOD Categories:** Encoded as `-1` by the preprocessor while preserving the raw string in explanation records.
   * **NumPy Scalar Types:** `np.str_`, `np.float32`, `np.int64` are coerced to standard native Python primitives for clean JSON serialization.

---

## 7. Reason Code Taxonomy & Synthesis

Reason codes are categorized by their origin to ensure investigators understand the source of each signal:

```
                               ┌─────────────────────────────────────────┐
                               │         Unified Reason Code List        │
                               └────────────────────┬────────────────────┘
                                                    │
                         ┌──────────────────────────┴──────────────────────────┐
                         ▼                                                     ▼
           ┌───────────────────────────┐                         ┌───────────────────────────┐
           │      source = "RULE"      │                         │      source = "MODEL"     │
           │  Deterministic Policies   │                         │  Continuous TreeSHAP Drivers│
           │  & Compliance Checks      │                         │  in Raw Margin Space      │
           └─────────────┬─────────────┘                         └─────────────┬─────────────┘
                         │                                                     │
                         ├─ Overriding Rules (High Severity)                   ├─ Risk Factors (ϕ > 0)
                         ├─ Actionable Rules (Medium Severity)                 └─ Mitigating Factors (ϕ < 0)
                         └─ Passive Monitors (Info Severity)
```

### 7.1 Reason Code Synthesis Rules

1. **Source Separation:**  
   Deterministic business rules are labeled `source="RULE"`. Continuous TreeSHAP attributions are labeled `source="MODEL"`. A model SHAP attribution is never mischaracterized as an independently validated business rule.
2. **Attribution Direction:**  
   * Positive attributions ($\phi_i > 0$) generate `RISK_INCREASING` model factors and reason codes.
   * Negative attributions ($\phi_i < 0$) generate `MITIGATING` model factors.
3. **Deterministic Ranking Hierarchy:**  
   * Rank 1..R: Overriding business rules (e.g. `RULE_VELOCITY_BURST_REVIEW`).
   * Next: Actionable non-overriding rules.
   * Next: Top model risk drivers sorted by attribution magnitude $|\phi_i|$ descending.
   * Last: Passive monitoring rules (e.g. `RULE_GEO_IMPOSSIBLE_TRAVEL_MONITOR`) with `severity="INFO"`.
4. **Deduplication:**  
   The synthesizer prevents duplicate codes from being emitted across multiple rule matches or feature drivers.

---

## 8. Empirical Performance Benchmarks & Methodology

> [!NOTE]
> Latency metrics represent benchmark observations under the local test environment and are not guaranteed production SLAs.

### 8.1 Benchmark Methodology
* **Operating Environment:** Windows 10 (Build 26200), Python 3.11.7 (64-bit AMD64), 16 Logical CPU Cores.
* **Libraries:** `xgboost==3.2.0`, `scikit-learn==1.3.2`, `pandas==2.1.3`, `numpy==1.26.2`.
* **Warm-up Protocol:** 50 unmeasured single-transaction warm-up iterations + 500 unmeasured batch rows.
* **Measured Single Iterations:** 1,000 independent single-transaction explanations evaluated across `val_features.parquet`.
* **Measured Batch Iterations:** 10 full iterations across $N = 10,000$ rows ($100,000$ cumulative rows).
* **Cold-Start Overhead:** Model loading measured separately.

### 8.2 Measured Latency & Throughput Results

| Evaluation Mode | Mean | Median | Min | Max | p90 | p95 | p99 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **End-to-End Single Explanation** | **$25.18\text{ ms}$** | $24.29\text{ ms}$ | $21.87\text{ ms}$ | $74.00\text{ ms}$ | $27.82\text{ ms}$ | $29.81\text{ ms}$ | $35.09\text{ ms}$ |
| **Batch TreeSHAP ($N=10,000$)** | **$602.77\ \mu\text{s/row}$** | $597.43\ \mu\text{s/row}$ | $579.39\ \mu\text{s/row}$ | $628.07\ \mu\text{s/row}$ | $625.50\ \mu\text{s/row}$ | $627.19\ \mu\text{s/row}$ | $627.90\ \mu\text{s/row}$ |

* **Single-Process Batch Throughput:** $\approx 1,659\text{ explanations/second}$.
* **Cold-Start Model Loading Overhead:** $154.51\text{ ms}$.

---

## 9. Verification, Test Evidence & Artifact Invariance

### 9.1 Test Execution Commands & Results

1. **Phase 7 Dedicated Tests:**
   ```bash
   python -m pytest tests/ml/test_explainability.py tests/ml/test_explainability_integration.py -v
   ```
   * **Result:** `26 passed in 8.57s` (100% pass rate).

2. **Phase 6 Risk Engine Independent Tests:**
   ```bash
   python -m pytest tests/ml/test_risk_engine.py tests/ml/test_risk_engine_integration.py tests/ml/test_risk_rules.py tests/ml/test_risk_catalog.py -q
   ```
   * **Result:** `250 passed in 8.33s` (100% pass rate).

3. **Complete Platform ML Test Suite:**
   ```bash
   python -m pytest tests/ml/ -q
   ```
   * **Result:** `358 passed in 46.87s` (100% pass rate across all 12 ML test modules).

4. **Bytecode Compilation Check:**
   ```bash
   python -m py_compile ml/explainability/schemas.py ml/explainability/config.py ml/explainability/explainer.py ml/explainability/reason_codes.py ml/explainability/__init__.py ml/risk_engine/evaluator.py tests/ml/test_explainability.py tests/ml/test_explainability_integration.py
   ```
   * **Result:** Exit code `0` (clean compilation).

### 9.2 Artifact SHA-256 Invariance
All frozen platform artifacts remain strictly unaltered:

| Artifact Path | SHA-256 Checksum | Invariance Status |
| :--- | :--- | :---: |
| `ml/models/artifacts/champion_model.joblib` | `5598fc3c9267c3fc14efe83cefd4678bb1fd42fef966200fccd69fd37508611d` | 🟢 Invariant |
| `ml/models/artifacts/champion_preprocessor.joblib` | `24f4783cdb0141ff13fb1f1036f6dc5d15c32002f915cb33964489c65594231b` | 🟢 Invariant |
| `ml/models/artifacts/model_metadata.json` | `deb0f0e8e8b5436fb38c84e0d9554de1f6ce8097b3b57af70aea02abefc35c27` | 🟢 Invariant |
| `ml/models/artifacts/all_models_validation_benchmark.json` | `021bb381bde732dbdd8602445479eab90887e32c75b21d07ae62f644e6d40fa8` | 🟢 Invariant |
| `ml/models/artifacts/threshold_analysis.json` | `dd95d5c0a6e666227c46159d8a2b2cea497f07e245a15556e2ff1afd76ad742b` | 🟢 Invariant |
| `ml/models/artifacts/feature_importance.json` | `9053e37e4cbb4d1427c920f5e928de18dbe4cf4c0b7e509104105f0e94d11c86` | 🟢 Invariant |
| `data/processed/features/val_features.parquet` | `999bdf4324810d9343a54beaa9f653da1ba4cd7430a51caaf19197afc45ffa92` | 🟢 Invariant |
| `data/processed/features/test_features.parquet` | `a03446cae9c9ca2dd5fed5265647a08ccf58f18999c7d4ec441845511ea83cf3` | 🟢 Invariant |

---

## 10. Operational Limitations & Governance Scope

1. **Investigative Support Scope:** The explainability module is designed to provide interpretability and triage assistance for fraud risk analysts. It does not provide statutory compliance certifications or autonomous legal defenses.
2. **Margin Space Additivity vs Non-Linear Probability:** TreeSHAP attributions are linear and additive in log-odds margin space. Because the logistic sigmoid function is non-linear, individual feature attributions cannot be linearly summed in probability space.
3. **Correlation vs Causation:** Local SHAP attributions quantify statistical contributions of features to the tree ensemble's margin score; they do not establish empirical causality or intent.
4. **Rule Threshold Calibration:** Deterministic business rule thresholds (e.g. 1-hour velocity $> 4$, Z-score $> 5\sigma$) reflect specific policy operating points and must be periodically reviewed against portfolio dynamics.
5. **Production Readiness Requirements:** Local benchmark observations do not constitute a high-availability production SLA. Real-world deployment requires integration into API serving layers (Phase 8), persistent storage (Phase 9), real-time benchmarking (Phase 10), and operational drift monitoring (Phase 13).
