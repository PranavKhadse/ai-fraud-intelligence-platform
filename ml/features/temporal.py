"""
Group 1: Temporal & Cyclic Feature Engineering Module.

Extracts leakage-free transaction-time features purely from current transaction timestamps.
"""

from typing import Dict
import pandas as pd
import numpy as np

from ml.features.config import TEMPORAL_FEATURES, NIGHT_HOURS


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 11 temporal and cyclic features from transaction timestamps.
    
    Features created:
    - transaction_hour (int32): 0..23
    - day_of_week (int32): 0=Monday..6=Sunday
    - day_of_month (int32): 1..31
    - month (int32): 1..12
    - week_of_year (int32): 1..53 (ISO week)
    - is_weekend (int8): 1 if Saturday or Sunday, else 0
    - is_night (int8): 1 if hour in [22, 23, 0, 1, 2, 3, 4], else 0 (consistent with Phase 2 EDA)
    - hour_sin (float32): sin(2 * pi * hour / 24)
    - hour_cos (float32): cos(2 * pi * hour / 24)
    - day_of_week_sin (float32): sin(2 * pi * dow / 7)
    - day_of_week_cos (float32): cos(2 * pi * dow / 7)
    """
    ts = pd.to_datetime(df["timestamp"])
    
    hour = ts.dt.hour.values.astype(np.int32)
    dow = ts.dt.dayofweek.values.astype(np.int32)
    dom = ts.dt.day.values.astype(np.int32)
    month = ts.dt.month.values.astype(np.int32)
    
    # ISO week number
    isocal = ts.dt.isocalendar()
    week_of_year = isocal.week.values.astype(np.int32)
    
    is_weekend = np.isin(dow, [5, 6]).astype(np.int8)
    is_night = np.isin(hour, NIGHT_HOURS).astype(np.int8)
    
    two_pi = 2.0 * np.pi
    hour_sin = np.sin(two_pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(two_pi * hour / 24.0).astype(np.float32)
    dow_sin = np.sin(two_pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(two_pi * dow / 7.0).astype(np.float32)
    
    out = pd.DataFrame(
        {
            "transaction_hour": hour,
            "day_of_week": dow,
            "day_of_month": dom,
            "month": month,
            "week_of_year": week_of_year,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week_sin": dow_sin,
            "day_of_week_cos": dow_cos,
        },
        index=df.index,
    )
    
    return out[TEMPORAL_FEATURES]
