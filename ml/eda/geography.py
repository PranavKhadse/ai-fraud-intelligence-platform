"""
Geographic analysis module.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Calculate the great-circle distance (in kilometers) between two geographic coordinates
    using the Haversine formula.
    
    R = 6,371.0 km (approximate Earth radius)
    """
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def analyze_geography(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze geographic patterns and cardholder-to-merchant distance for legitimate vs fraud.
    """
    distances = haversine_distance_km(
        df["cardholder_lat"].values,
        df["cardholder_long"].values,
        df["merchant_lat"].values,
        df["merchant_long"].values,
    )

    df_geo = df[["is_fraud"]].copy()
    df_geo["distance_km"] = distances

    legit_dist = df_geo[df_geo["is_fraud"] == 0]["distance_km"]
    fraud_dist = df_geo[df_geo["is_fraud"] == 1]["distance_km"]

    def dist_stats(s: pd.Series) -> Dict[str, float]:
        return {
            "mean_km": float(s.mean()),
            "std_km": float(s.std()),
            "median_km": float(s.median()),
            "min_km": float(s.min()),
            "max_km": float(s.max()),
            "p25_km": float(s.quantile(0.25)),
            "p50_km": float(s.quantile(0.50)),
            "p75_km": float(s.quantile(0.75)),
            "p90_km": float(s.quantile(0.90)),
            "p99_km": float(s.quantile(0.99)),
        }

    return {
        "cardholder_lat_range": [float(df["cardholder_lat"].min()), float(df["cardholder_lat"].max())],
        "cardholder_long_range": [float(df["cardholder_long"].min()), float(df["cardholder_long"].max())],
        "merchant_lat_range": [float(df["merchant_lat"].min()), float(df["merchant_lat"].max())],
        "merchant_long_range": [float(df["merchant_long"].min()), float(df["merchant_long"].max())],
        "distance_all_transactions": dist_stats(df_geo["distance_km"]),
        "distance_legitimate": dist_stats(legit_dist),
        "distance_fraudulent": dist_stats(fraud_dist),
    }
