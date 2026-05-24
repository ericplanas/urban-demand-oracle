"""
transform.py — Build the demand_hourly aggregation from clean trip data.

The ML model does NOT train on raw trips. It trains on demand_hourly:
    (zone_id, hour_start) → trip_count

This is the key modelling decision: we're forecasting demand (counts),
not individual trip attributes. Aggregating here makes training ~1000x faster
and the target variable meaningful.

Also extracts time features used as ML features:
    hour_of_day, day_of_week, is_weekend, month
"""
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [transform] %(message)s")
log = logging.getLogger(__name__)


def build_demand_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clean trips into hourly demand per pickup zone.
    Returns a DataFrame with one row per (zone_id, hour_start).
    """
    df = df.copy()

    # Floor pickup_datetime to the hour — this is our time bucket
    df["hour_start"] = df["pickup_datetime"].dt.floor("h")

    agg = (
        df.groupby(["pickup_zone_id", "hour_start"])
        .agg(
            trip_count=("pickup_zone_id", "count"),
            avg_fare=("fare_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
        )
        .reset_index()
        .rename(columns={"pickup_zone_id": "zone_id"})
    )

    # Time features — these become ML model features in Step 5
    agg["hour_of_day"] = agg["hour_start"].dt.hour
    agg["day_of_week"] = agg["hour_start"].dt.dayofweek  # 0=Monday
    agg["is_weekend"] = (agg["day_of_week"] >= 5).astype(int)
    agg["month"] = agg["hour_start"].dt.month

    log.info(f"  demand_hourly rows built: {len(agg):,}")
    return agg


if __name__ == "__main__":
    # Quick smoke test with fake data
    import numpy as np
    fake = pd.DataFrame({
        "pickup_datetime": pd.date_range("2023-01-01", periods=1000, freq="5min"),
        "pickup_zone_id": np.random.randint(1, 10, 1000),
        "fare_amount": np.random.uniform(5, 50, 1000),
        "trip_distance": np.random.uniform(0.5, 10, 1000),
    })
    result = build_demand_hourly(fake)
    print(result.head(10))
