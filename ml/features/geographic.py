"""
Group 7: Geographic & Travel Velocity Feature Engineering Module.

Computes Haversine distances between cardholder and merchant coordinates,
displacement distance between consecutive transactions for the same account (with strictly earlier timestamp),
implied travel speed (km/h), and impossible travel speed indicator (>800 km/h).
"""

from typing import Dict
import pandas as pd
import numpy as np

from ml.features.config import (
    GEOGRAPHIC_FEATURES,
    EARTH_RADIUS_KM,
    IMPOSSIBLE_SPEED_THRESHOLD_KMH,
)


def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized Haversine distance in kilometers between pairs of (lat, lon) coordinates.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    # Numerical clip for asin/arctan domain stability
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def compute_geographic_features_for_account(
    unix_times: np.ndarray,
    merch_lats: np.ndarray,
    merch_longs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Vectorized calculation of previous merchant distance, implied travel speed,
    and impossible travel indicator for a single account's transactions.
    
    Args:
        unix_times: 1D int64 array of unix timestamps (sorted non-decreasingly).
        merch_lats: 1D float64 array of merchant latitudes.
        merch_longs: 1D float64 array of merchant longitudes.
        
    Returns:
        Dict mapping feature names to 1D numpy arrays.
    """
    m = len(unix_times)
    if m == 0:
        return {
            "distance_from_prev_merchant_km": np.empty(0, dtype=np.float32),
            "implied_travel_speed_kmh": np.empty(0, dtype=np.float32),
            "is_impossible_travel_speed": np.empty(0, dtype=np.int8),
        }

    # Strict point-in-time index
    idx_strict = np.searchsorted(unix_times, unix_times, side="left")
    prev_idx = idx_strict - 1
    has_prev = prev_idx >= 0
    
    dist_from_prev = np.zeros(m, dtype=np.float32)
    speed_kmh = np.zeros(m, dtype=np.float32)
    
    if np.any(has_prev):
        curr_lats = merch_lats[has_prev]
        curr_longs = merch_longs[has_prev]
        p_idx = prev_idx[has_prev]
        prev_lats = merch_lats[p_idx]
        prev_longs = merch_longs[p_idx]
        
        dists = haversine_distance_km(prev_lats, prev_longs, curr_lats, curr_longs)
        dist_from_prev[has_prev] = dists.astype(np.float32)
        
        dt_seconds = (unix_times[has_prev] - unix_times[p_idx]).astype(np.float64)
        valid_dt = dt_seconds > 0
        
        if np.any(valid_dt):
            # Speed in km/h = distance_km / (dt_seconds / 3600.0)
            speeds = dists[valid_dt] / (dt_seconds[valid_dt] / 3600.0)
            
            # Place speeds into valid indices
            valid_indices = np.where(has_prev)[0][valid_dt]
            speed_kmh[valid_indices] = speeds.astype(np.float32)
            
    is_impossible = (speed_kmh > IMPOSSIBLE_SPEED_THRESHOLD_KMH).astype(np.int8)
    
    return {
        "distance_from_prev_merchant_km": dist_from_prev,
        "implied_travel_speed_kmh": speed_kmh,
        "is_impossible_travel_speed": is_impossible,
    }


def extract_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 4 geographic & travel velocity features:
    - cardholder_merchant_distance_km (float32)
    - distance_from_prev_merchant_km (float32)
    - implied_travel_speed_kmh (float32)
    - is_impossible_travel_speed (int8)
    """
    n = len(df)
    
    # 1. Cardholder to Merchant static Haversine distance
    card_merch_dist = haversine_distance_km(
        df["cardholder_lat"].values,
        df["cardholder_long"].values,
        df["merchant_lat"].values,
        df["merchant_long"].values,
    ).astype(np.float32)
    
    # 2. Account-level sequential displacement
    dist_from_prev = np.zeros(n, dtype=np.float32)
    speed_kmh = np.zeros(n, dtype=np.float32)
    is_impossible = np.zeros(n, dtype=np.int8)
    
    grouped = df.groupby("account_id", sort=False)
    for _, group in grouped:
        indices = group.index.values
        u_times = group["unix_time"].values
        lats = group["merchant_lat"].values
        longs = group["merchant_long"].values
        
        geo_feats = compute_geographic_features_for_account(u_times, lats, longs)
        dist_from_prev[indices] = geo_feats["distance_from_prev_merchant_km"]
        speed_kmh[indices] = geo_feats["implied_travel_speed_kmh"]
        is_impossible[indices] = geo_feats["is_impossible_travel_speed"]
        
    out = pd.DataFrame(
        {
            "cardholder_merchant_distance_km": card_merch_dist,
            "distance_from_prev_merchant_km": dist_from_prev,
            "implied_travel_speed_kmh": speed_kmh,
            "is_impossible_travel_speed": is_impossible,
        },
        index=df.index,
    )
    
    return out[GEOGRAPHIC_FEATURES]
