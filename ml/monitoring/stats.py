"""
Statistical Algorithms & Drift Metrics for ML & Model Monitoring.

Provides vectorized and mathematically robust implementations for:
- Population Stability Index (PSI) for binned numerical & discrete distributions.
- Two-sample Kolmogorov-Smirnov (KS) test with asymptotic p-value.
- Jensen-Shannon Divergence (JSD) for categorical distributions.
- Missing-rate delta and unseen-category rate calculation.
"""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def calculate_psi(
    expected_proportions: Sequence[float],
    observed_proportions: Sequence[float],
    epsilon: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index (PSI) between expected and observed binned proportions.

    Formula:
        PSI = sum_{i=1}^k (P_obs,i - P_exp,i) * ln((P_obs,i + eps) / (P_exp,i + eps))

    Args:
        expected_proportions: Sequence of baseline/expected bucket probabilities.
        observed_proportions: Sequence of production/observed bucket probabilities.
        epsilon: Laplace smoothing constant to prevent division by zero or log(0).

    Returns:
        float: Non-negative PSI value. Returns 0.0 if either input is empty.
    """
    p_exp = np.asarray(expected_proportions, dtype=np.float64)
    p_obs = np.asarray(observed_proportions, dtype=np.float64)

    if len(p_exp) == 0 or len(p_obs) == 0:
        return 0.0

    if len(p_exp) != len(p_obs):
        raise ValueError(
            f"Expected and observed proportions must have the same length: {len(p_exp)} != {len(p_obs)}"
        )

    # Normalize to ensure proper probability distributions if sums > 0
    exp_sum = np.sum(p_exp)
    obs_sum = np.sum(p_obs)
    if exp_sum > 0:
        p_exp = p_exp / exp_sum
    if obs_sum > 0:
        p_obs = p_obs / obs_sum

    psi_contributions = (p_obs - p_exp) * np.log((p_obs + epsilon) / (p_exp + epsilon))
    psi_val = float(np.sum(psi_contributions))
    return max(0.0, psi_val)


def calculate_numerical_psi(
    observed_values: Union[np.ndarray, Sequence[float], pd.Series],
    baseline_bin_edges: Sequence[float],
    baseline_expected_proportions: Optional[Sequence[float]] = None,
    baseline_ref_samples: Optional[Sequence[float]] = None,
    epsilon: float = 1e-4,
) -> Tuple[float, List[float], List[float]]:
    """
    Calculate PSI for continuous numerical observations against baseline decile bin edges.

    Uses `np.digitize` on the interior bin boundaries to guarantee that all values
    in (-inf, +inf) are partitioned across the exact K bins.

    Args:
        observed_values: 1D array-like of continuous numerical values.
        baseline_bin_edges: K+1 quantile bin boundaries (e.g. 11 boundaries for 10 deciles).
        baseline_expected_proportions: Optional precomputed expected probabilities per bin.
        baseline_ref_samples: Optional reference samples to compute baseline bin proportions.
        epsilon: Laplace smoothing constant.

    Returns:
        Tuple of (psi_value, observed_proportions, expected_proportions).
    """
    if isinstance(observed_values, pd.Series):
        observed_clean = observed_values.dropna().to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(observed_values, dtype=np.float64)
        observed_clean = arr[~np.isnan(arr) & ~np.isinf(arr)]

    edges = np.asarray(baseline_bin_edges, dtype=np.float64)
    if len(edges) < 2:
        raise ValueError("baseline_bin_edges must contain at least 2 boundaries.")

    num_bins = len(edges) - 1
    cutoffs = edges[1:-1]  # interior cutoffs

    # Determine baseline expected proportions
    if baseline_expected_proportions is not None and len(baseline_expected_proportions) == num_bins:
        exp_props = np.asarray(baseline_expected_proportions, dtype=np.float64)
        exp_sum = np.sum(exp_props)
        if exp_sum > 0:
            exp_props = exp_props / exp_sum
    elif baseline_ref_samples is not None and len(baseline_ref_samples) > 0:
        ref_arr = np.asarray(baseline_ref_samples, dtype=np.float64)
        ref_clean = ref_arr[~np.isnan(ref_arr) & ~np.isinf(ref_arr)]
        if len(ref_clean) > 0 and len(cutoffs) > 0:
            ref_idx = np.digitize(ref_clean, cutoffs, right=False)
            ref_counts = np.bincount(ref_idx, minlength=num_bins)
            exp_props = ref_counts / len(ref_clean)
        else:
            exp_props = np.full(num_bins, 1.0 / num_bins, dtype=np.float64)
    else:
        exp_props = np.full(num_bins, 1.0 / num_bins, dtype=np.float64)

    if len(observed_clean) == 0:
        obs_props = np.zeros(num_bins, dtype=np.float64)
        return 0.0, obs_props.tolist(), exp_props.tolist()

    if len(cutoffs) > 0:
        obs_idx = np.digitize(observed_clean, cutoffs, right=False)
        obs_counts = np.bincount(obs_idx, minlength=num_bins)
        obs_props = obs_counts / len(observed_clean)
    else:
        obs_props = np.ones(num_bins, dtype=np.float64)

    psi_val = calculate_psi(exp_props, obs_props, epsilon=epsilon)
    return psi_val, obs_props.tolist(), exp_props.tolist()


def calculate_two_sample_ks(
    observed_values: Union[np.ndarray, Sequence[float], pd.Series],
    reference_samples: Union[np.ndarray, Sequence[float], pd.Series],
    method: str = "asymp",
) -> Tuple[float, float]:
    """
    Compute two-sample Kolmogorov-Smirnov statistic and asymptotic p-value.

    Formula:
        D = sup_x |F_obs(x) - F_ref(x)|

    Args:
        observed_values: 1D array of observed production values.
        reference_samples: 1D array of frozen reference baseline observations (e.g. 5,000 samples).
        method: KS p-value calculation method (default 'asymp' for asymptotic distribution).

    Returns:
        Tuple of (ks_statistic, asymptotic_p_value).
    """
    if isinstance(observed_values, pd.Series):
        obs_clean = observed_values.dropna().to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(observed_values, dtype=np.float64)
        obs_clean = arr[~np.isnan(arr) & ~np.isinf(arr)]

    if isinstance(reference_samples, pd.Series):
        ref_clean = reference_samples.dropna().to_numpy(dtype=np.float64)
    else:
        arr_ref = np.asarray(reference_samples, dtype=np.float64)
        ref_clean = arr_ref[~np.isnan(arr_ref) & ~np.isinf(arr_ref)]

    if len(obs_clean) == 0 or len(ref_clean) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(obs_clean, ref_clean, method=method)
    return float(res.statistic), float(res.pvalue)


def calculate_jsd(
    p_dict: Dict[str, float],
    q_dict: Dict[str, float],
    epsilon: float = 1e-6,
) -> float:
    """
    Compute base-2 Jensen-Shannon Divergence (JSD in [0.0, 1.0]) between two discrete distributions.

    Formula:
        JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
        where M = 0.5 * (P + Q) and D_KL uses log2.

    Args:
        p_dict: Baseline discrete probability distribution {category: probability}.
        q_dict: Observed discrete probability distribution {category: probability}.
        epsilon: Floor for zero probabilities in divergence calculation.

    Returns:
        float: Bounded divergence in [0.0, 1.0].
    """
    all_keys = sorted(set(p_dict.keys()) | set(q_dict.keys()))
    if not all_keys:
        return 0.0

    p_vec = np.array([max(0.0, float(p_dict.get(k, 0.0))) for k in all_keys], dtype=np.float64)
    q_vec = np.array([max(0.0, float(q_dict.get(k, 0.0))) for k in all_keys], dtype=np.float64)

    p_sum = np.sum(p_vec)
    q_sum = np.sum(q_vec)

    if p_sum > 0:
        p_vec = p_vec / p_sum
    if q_sum > 0:
        q_vec = q_vec / q_sum

    if p_sum == 0 and q_sum == 0:
        return 0.0
    if p_sum == 0 or q_sum == 0:
        return 1.0

    m_vec = 0.5 * (p_vec + q_vec)

    # Compute D_KL(P || M) using log2
    kl_p_m = 0.0
    p_mask = p_vec > 0
    if np.any(p_mask):
        kl_p_m = np.sum(p_vec[p_mask] * np.log2(p_vec[p_mask] / m_vec[p_mask]))

    # Compute D_KL(Q || M) using log2
    kl_q_m = 0.0
    q_mask = q_vec > 0
    if np.any(q_mask):
        kl_q_m = np.sum(q_vec[q_mask] * np.log2(q_vec[q_mask] / m_vec[q_mask]))

    jsd_val = 0.5 * kl_p_m + 0.5 * kl_q_m
    return float(np.clip(jsd_val, 0.0, 1.0))


def calculate_missing_rate_delta(
    observed_missing_rate: float,
    baseline_missing_rate: float,
) -> float:
    """
    Calculate absolute missing-rate delta between observed window and baseline.

    Formula:
        delta_missing = |rate_prod - rate_base|
    """
    return float(abs(float(observed_missing_rate) - float(baseline_missing_rate)))


def calculate_unseen_category_rate(
    observed_categories: Union[pd.Series, Sequence[Any], np.ndarray],
    baseline_vocabulary: Sequence[str],
) -> Tuple[float, List[str]]:
    """
    Calculate the proportion of observed observations that belong to categories
    not present in the baseline vocabulary.

    Args:
        observed_categories: 1D sequence of categorical labels.
        baseline_vocabulary: Known baseline category labels.

    Returns:
        Tuple of (unseen_category_rate, list_of_unseen_categories).
    """
    if isinstance(observed_categories, pd.Series):
        obs_clean = observed_categories.dropna().astype(str).tolist()
    else:
        obs_clean = [str(x) for x in observed_categories if x is not None and not (isinstance(x, float) and np.isnan(x))]

    if len(obs_clean) == 0:
        return 0.0, []

    base_vocab_set = set(str(k) for k in baseline_vocabulary)
    unseen_items = [c for c in obs_clean if c not in base_vocab_set]
    unseen_distinct = sorted(list(set(unseen_items)))
    unseen_rate = len(unseen_items) / len(obs_clean)

    return float(unseen_rate), unseen_distinct
