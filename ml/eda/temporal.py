"""
Temporal fraud analysis module.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def analyze_temporal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze fraud distribution across temporal dimensions (hour, day of week, month, hour x day-of-week).
    """
    df_temp = df.copy()
    ts = pd.to_datetime(df_temp["timestamp"])
    df_temp["hour"] = ts.dt.hour
    df_temp["day_of_week"] = ts.dt.day_name()
    df_temp["day_of_week_num"] = ts.dt.dayofweek  # 0=Monday, 6=Sunday
    df_temp["year_month"] = ts.dt.to_period("M").astype(str)

    # 1. Hourly aggregation
    hourly = (
        df_temp.groupby("hour")
        .agg(
            total_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: float(x.mean() * 100)),
        )
        .reset_index()
    )

    # 2. Day-of-week aggregation
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = (
        df_temp.groupby("day_of_week")
        .agg(
            total_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: float(x.mean() * 100)),
        )
        .reindex(dow_order)
        .reset_index()
    )

    # 3. Hour x Day-of-Week 2D Matrix
    hour_dow_pivot = df_temp.pivot_table(
        index="day_of_week",
        columns="hour",
        values="is_fraud",
        aggfunc=lambda x: float(np.mean(x) * 100),
    ).reindex(dow_order)

    # 4. Nighttime vs Daytime comparison
    # Nighttime: 22:00 to 04:00, Daytime: 05:00 to 21:00
    df_temp["is_night"] = df_temp["hour"].isin([22, 23, 0, 1, 2, 3, 4])
    night_stats = (
        df_temp.groupby("is_night")
        .agg(
            total_count=("is_fraud", "count"),
            fraud_count=("is_fraud", "sum"),
            fraud_rate=("is_fraud", lambda x: float(x.mean() * 100)),
        )
        .to_dict(orient="index")
    )

    # Peak hour and day identification
    peak_fraud_hour = int(hourly.loc[hourly["fraud_rate"].idxmax()]["hour"])
    peak_fraud_hour_rate = float(hourly.loc[hourly["fraud_rate"].idxmax()]["fraud_rate"])
    lowest_fraud_hour = int(hourly.loc[hourly["fraud_rate"].idxmin()]["hour"])
    lowest_fraud_hour_rate = float(hourly.loc[hourly["fraud_rate"].idxmin()]["fraud_rate"])

    peak_vol_hour = int(hourly.loc[hourly["total_count"].idxmax()]["hour"])
    peak_vol_hour_count = int(hourly.loc[hourly["total_count"].idxmax()]["total_count"])

    return {
        "hourly_stats": hourly.to_dict(orient="records"),
        "dow_stats": dow.to_dict(orient="records"),
        "hour_dow_matrix": hour_dow_pivot.to_dict(),
        "night_vs_day": {
            "night": night_stats.get(True, {}),
            "day": night_stats.get(False, {}),
        },
        "insights": {
            "peak_fraud_rate_hour": peak_fraud_hour,
            "peak_fraud_rate_value": round(peak_fraud_hour_rate, 3),
            "lowest_fraud_rate_hour": lowest_fraud_hour,
            "lowest_fraud_rate_value": round(lowest_fraud_hour_rate, 3),
            "peak_volume_hour": peak_vol_hour,
            "peak_volume_count": peak_vol_hour_count,
        },
    }
