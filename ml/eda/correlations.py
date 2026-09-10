"""
Correlation and association analysis module.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from scipy import stats


def cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Calculate Cramér's V statistic for categorical association.
    """
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def analyze_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Pearson, Spearman correlations for numeric columns and Cramér's V for categoricals.
    """
    num_cols = ["amount", "unix_time", "city_pop", "cardholder_lat", "cardholder_long", "merchant_lat", "merchant_long", "is_fraud"]
    df_num = df[num_cols].dropna()

    pearson_corr = df_num.corr(method="pearson").round(4).to_dict()
    spearman_corr = df_num.corr(method="spearman").round(4).to_dict()

    # Categorical associations with is_fraud
    cat_cols = ["merchant_category", "job_category"]
    cat_associations: Dict[str, float] = {}

    for col in cat_cols:
        contingency = pd.crosstab(df[col], df["is_fraud"])
        cv = cramers_v(contingency)
        cat_associations[col] = round(cv, 4)

    return {
        "pearson_matrix": pearson_corr,
        "spearman_matrix": spearman_corr,
        "numeric_fraud_correlations": {
            "pearson": {k: v["is_fraud"] for k, v in pearson_corr.items() if k != "is_fraud"},
            "spearman": {k: v["is_fraud"] for k, v in spearman_corr.items() if k != "is_fraud"},
        },
        "categorical_fraud_associations_cramers_v": cat_associations,
    }
