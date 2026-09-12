# Phase 5: Probability Calibration Feasibility & Leakage Governance Report

**Platform:** AI-Powered Fraud Detection & Risk Intelligence Platform
**Phase:** Phase 5 (Imbalance Handling & Cost Optimization — Calibration Feasibility Investigation)
**Status:** Completed & Documented
**Champion Architecture:** XGBoost (`champion_model.joblib`)

---

## 1. Executive Summary & Objective

The objective of this investigation is to evaluate whether probability calibration (e.g., Platt Sigmoid Scaling or Isotonic Regression) can be performed safely and without data leakage on the existing Phase 4 XGBoost champion model, given the repository's chronological train/validation/test split methodology.

### Key Finding:
1. **Raw XGBoost Outputs are Uncalibrated Model Scores**: Because XGBoost was trained with `scale_pos_weight = 171.75` to penalize false negatives under extreme class imbalance (~0.58% fraud rate), the raw model scores are shifted significantly upwards. A score of $0.55$ corresponds to an empirical fraud probability of only $\approx 4.96\%$.
2. **Current Decision Optimization Layer is Ranking-Optimal**: Decision threshold cost optimization operates directly on score ordering (PR-AUC = 0.9619). Monotonic probability transformations do not alter ranking or change the optimal decision boundary ($\tau^* = 0.78$).
3. **No Zero-Leakage Holdout Currently Exists for Calibration Without Retraining**:
   - The training set was fully used for model fitting (in-sample predictions are overconfident).
   - The validation set was used for model selection and threshold search (fitting a calibrator on validation creates in-sample calibration bias).
   - The out-of-time (OOT) test set is strictly protected and must never be accessed for calibration.
4. **Recommendation**: Preserve the fixed-cost threshold decision layer on model scores for Phase 5. Introduce a chronological calibration split (Train-Fit $\to$ Train-Calibrate $\to$ Validation $\to$ Test) in the retraining pipeline (Phase 14).

---

## 2. Empirical Calibration Diagnosis of Raw Model Scores

To quantify the distortion introduced by `scale_pos_weight = 171.75`, we evaluated the reliability curve and Brier score of raw validation model scores against ground-truth labels:

### Reliability Breakdown (10 Uniform Bins on Validation Set):

| Score Bin Range | Mean Predicted Score | Empirical True Fraud Rate | Distortion Factor (Predicted / True) |
| :--- | :--- | :--- | :--- |
| **[0.00, 0.10)** | 0.0020 | 0.0100% (0.0001) | $20.0\times$ over-estimated |
| **[0.10, 0.20)** | 0.1411 | 0.9100% (0.0091) | $15.5\times$ over-estimated |
| **[0.20, 0.30)** | 0.2457 | 1.0400% (0.0104) | $23.6\times$ over-estimated |
| **[0.30, 0.40)** | 0.3475 | 2.4200% (0.0242) | $14.4\times$ over-estimated |
| **[0.40, 0.50)** | 0.4475 | 3.1600% (0.0316) | $14.2\times$ over-estimated |
| **[0.50, 0.60)** | 0.5502 | 4.9600% (0.0496) | $11.1\times$ over-estimated |
| **[0.60, 0.70)** | 0.6486 | 3.9000% (0.0390) | $16.6\times$ over-estimated |
| **[0.70, 0.80)** | 0.7466 | 8.9200% (0.0892) | $8.4\times$ over-estimated |
| **[0.80, 0.90)** | 0.8500 | 18.4800% (0.1848) | $4.6\times$ over-estimated |
| **[0.90, 1.00]** | 0.9888 | 89.1600% (0.8916) | $1.1\times$ calibrated |

- **Brier Score (Raw Scores)**: `0.002043`
- **Takeaway**: Raw XGBoost outputs represent monotonic ranking scores, **not true posterior probabilities $P(\text{Fraud}|X)$**.

---

## 3. Data Split & Leakage Governance Audit

| Partition | Row Count | Fraud Count | Time Window | Role in Phase 1–5 | Calibration Eligibility | Leakage Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 1,296,675 | 7,506 (0.5789%) | 2019-01-01 to 2020-06-21 | Model training & encoder fitting | ❌ Ineligible (In-sample overfit) | Evaluating calibrator on training scores produces artificial overconfidence. |
| **Validation** | 277,859 | 1,221 (0.4394%) | 2020-06-21 to 2020-10-03 | Model selection & threshold search ($\tau^*=0.78$) | ⚠️ Exploratory Only | Fitting on validation causes in-sample evaluation bias; no separate validation holdout remains. |
| **OOT Test** | 277,860 | 924 (0.3325%) | 2020-10-03 to 2020-12-31 | Final frozen benchmark | ⛔ STRICTLY FORBIDDEN | Must NEVER be used for calibration fitting or selection. |

---

## 4. Calibration Architecture Options

### Option 1: Direct Threshold Decision Layer on Model Scores (Current Phase 5 — Recommended)
- **Mechanism**: Optimize the decision boundary $\tau^* = 0.78$ directly on the model score distribution.
- **Advantages**:
  - Requires zero retraining of Phase 4 models.
  - Zero data leakage or temporal partition contamination.
  - Ranking discrimination ($\text{PR-AUC} = 0.9619$) is mathematically preserved.
  - Minimum expected cost ($\$16,910.00$) is achieved identically because any strictly monotonic calibrator $g(s)$ simply maps the threshold to $\tau_{\text{cal}} = g(\tau^*)$ without changing classification outcomes.
- **Limitation**: Model scores cannot be interpreted directly as percentage fraud probabilities (e.g., in a UI displaying "78% probability of fraud").

---

### Option 2: Chronological Train Split Carve-Out (Future Retraining — Phase 14)
- **Mechanism**: Partition the 18-month Train set chronologically into:
  1. `Train-Fit` (Jan 1, 2019 to Feb 29, 2020 — ~1,000,000 rows): Train XGBoost trees.
  2. `Train-Calibrate` (Mar 1, 2020 to Jun 21, 2020 — ~296,675 rows): Fit Platt / Isotonic calibrator.
  3. `Validation` (Jun 21, 2020 to Oct 3, 2020): Evaluate calibrated probabilities and select threshold.
  4. `OOT Test` (Oct 3, 2020 to Dec 31, 2020): Single final frozen evaluation.
- **Advantages**: Completely leakage-free, temporally sound, provides genuine posterior probabilities for Phase 6 (0–100 risk scoring).
- **Trade-off**: Requires retraining the champion model on the reduced `Train-Fit` split.

---

### Option 3: Out-of-Fold (OOF) K-Fold Calibration During Training
- **Mechanism**: Train 5-fold cross-validated XGBoost models on the Train partition. Collect out-of-fold validation scores for all 1.29M rows. Fit a single Platt calibrator on the concatenated OOF scores.
- **Advantages**: Retains 100% of training data for tree learning.
- **Trade-off**: Requires running 5 separate XGBoost training jobs (~2.5 minutes).

---

## 5. Summary Recommendation for Phase 5 & Downstream Phases

1. **Phase 5 Decision**: Keep Phase 5 focused on the robust, leakage-free threshold optimization framework operating on model scores.
2. **Phase 6 Alignment**: In Phase 6 (Risk Engine & 0–100 Composite Scoring), linear/percentile risk mapping on model scores preserves risk ordering without violating temporal split boundaries.
3. **Phase 14 Retraining**: Full probability calibration pipelines with chronological carve-outs will be integrated as part of automated model retraining.
