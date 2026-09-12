"""
Risk Score Normalization & Semantic Risk Tier Mapping.

Converts continuous model ranking scores in [0.0, 1.0] into standardized integer
Risk Scores in [0, 100] and maps risk scores to semantic risk tiers.

IMPORTANT:
- The raw model scores from XGBoost reflect class imbalance adjustments (scale_pos_weight = 171.75)
  and represent continuous ranking scores, NOT certified empirical posterior probabilities.
- The normalized risk score is a standardized operational rating for risk triage, not a calibrated probability.
"""

import math
from typing import Union
import numpy as np

from ml.risk_engine.config import RiskTier


def normalize_model_score(model_score: Union[float, int, np.floating, np.integer]) -> int:
    """
    Convert a single raw model ranking score in [0.0, 1.0] to an integer Risk Score in [0, 100].

    Formula:
        Risk Score = clip(round(100 * model_score), 0, 100)

    Validation & Error Handling:
        - Rejects boolean types (which inherit from int).
        - Rejects non-numeric types (e.g. str, None, list).
        - Rejects non-finite values (NaN, +inf, -inf).
        - Rejects values outside the valid range [0.0, 1.0].

    Args:
        model_score: Raw model output score in [0.0, 1.0].

    Returns:
        int: Normalized integer risk score between 0 and 100 inclusive.

    Raises:
        TypeError: If input is boolean or non-numeric.
        ValueError: If input is NaN, infinite, or outside [0.0, 1.0].
    """
    # 1. Reject boolean inputs
    if isinstance(model_score, bool):
        raise TypeError(f"model_score cannot be a boolean value, got {model_score!r}")

    # 2. Type check
    if not isinstance(model_score, (int, float, np.floating, np.integer)):
        raise TypeError(
            f"model_score must be numeric (float or int), got {type(model_score).__name__}"
        )

    # 3. Finiteness check
    val = float(model_score)
    if not math.isfinite(val):
        raise ValueError(f"model_score must be a finite number, got {val}")

    # 4. Strict range check [0.0, 1.0]
    if val < 0.0 or val > 1.0:
        raise ValueError(f"model_score must be within [0.0, 1.0], got {val}")

    # 5. Deterministic round and clip
    rounded = round(100.0 * val)
    clipped = max(0, min(100, int(rounded)))
    return int(clipped)


def normalize_model_scores(model_scores: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion of a 1D NumPy array of raw model ranking scores to integer Risk Scores in [0, 100].

    Rounding Specification:
        Uses NumPy's `np.rint` which executes IEEE 754 round-half-to-even (matching Python 3's built-in `round()`),
        followed by clipping to [0, 100] and casting to `np.int64`.

    Validation & Error Handling:
        - Must be a NumPy ndarray.
        - Rejects boolean arrays.
        - Must be exactly 1-dimensional.
        - Must not be empty.
        - Must be a numeric dtype (floating or integer).
        - Rejects arrays containing NaN or +/-inf.
        - Rejects arrays with any element outside [0.0, 1.0].

    Args:
        model_scores: 1D NumPy array of raw model scores in [0.0, 1.0].

    Returns:
        np.ndarray: 1D array of dtype `np.int64` containing integer risk scores in [0, 100].

    Raises:
        TypeError: If input is not a numeric NumPy array or is a boolean array.
        ValueError: If array is empty, multidimensional, contains NaN/inf, or has out-of-range elements.
    """
    # 1. Must be a numpy ndarray
    if not isinstance(model_scores, np.ndarray):
        raise TypeError(
            f"model_scores must be a numpy.ndarray, got {type(model_scores).__name__}"
        )

    # 2. Reject boolean arrays
    if model_scores.dtype == bool or np.issubdtype(model_scores.dtype, np.bool_):
        raise TypeError("model_scores cannot be a boolean array.")

    # 3. Dimensionality check (must be 1D)
    if model_scores.ndim != 1:
        raise ValueError(
            f"model_scores must be a 1-dimensional array, got shape {model_scores.shape} (ndim={model_scores.ndim})"
        )

    # 4. Empty check
    if model_scores.size == 0:
        raise ValueError("model_scores array cannot be empty.")

    # 5. Numeric dtype check
    if not np.issubdtype(model_scores.dtype, np.number):
        raise TypeError(
            f"model_scores must have a numeric dtype, got dtype={model_scores.dtype}"
        )

    # 6. Finiteness check
    if not np.all(np.isfinite(model_scores)):
        raise ValueError("model_scores contains NaN or infinite values.")

    # 7. Range check [0.0, 1.0]
    if np.any((model_scores < 0.0) | (model_scores > 1.0)):
        min_v = float(np.min(model_scores))
        max_v = float(np.max(model_scores))
        raise ValueError(
            f"All model_scores elements must be within [0.0, 1.0]. Found values in [{min_v}, {max_v}]."
        )

    # 8. Deterministic round, clip, and cast
    scores_f64 = model_scores.astype(np.float64, copy=False)
    rounded = np.rint(scores_f64 * 100.0)
    clipped = np.clip(rounded, 0.0, 100.0)
    return clipped.astype(np.int64)


def risk_tier_from_score(risk_score: Union[int, np.integer]) -> RiskTier:
    """
    Map an integer Risk Score in [0, 100] to its corresponding semantic RiskTier.

    Bands:
    - 0  <= risk_score < 35  -> RiskTier.LOW
    - 35 <= risk_score < 60  -> RiskTier.MEDIUM
    - 60 <= risk_score < 78  -> RiskTier.HIGH
    - 78 <= risk_score <= 100 -> RiskTier.CRITICAL

    Validation & Error Handling:
        - Rejects boolean types.
        - Rejects non-integer types (e.g. float 35.0 is rejected to prevent silent precision errors).
        - Rejects values outside the range [0, 100].

    Args:
        risk_score: Integer risk score in [0, 100].

    Returns:
        RiskTier: Corresponding semantic risk tier.

    Raises:
        TypeError: If input is a boolean or not an integer.
        ValueError: If risk_score is outside [0, 100].
    """
    # 1. Reject boolean inputs
    if isinstance(risk_score, bool):
        raise TypeError(f"risk_score cannot be a boolean value, got {risk_score!r}")

    # 2. Must be an integer type
    if not isinstance(risk_score, (int, np.integer)):
        raise TypeError(
            f"risk_score must be an integer, got {type(risk_score).__name__} ({risk_score!r})"
        )

    # 3. Range check [0, 100]
    score_int = int(risk_score)
    if score_int < 0 or score_int > 100:
        raise ValueError(f"risk_score must be an integer between 0 and 100, got {score_int}")

    # 4. Exact boundary evaluation
    if score_int < 35:
        return RiskTier.LOW
    elif score_int < 60:
        return RiskTier.MEDIUM
    elif score_int < 78:
        return RiskTier.HIGH
    else:
        return RiskTier.CRITICAL
