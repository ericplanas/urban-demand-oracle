"""
Feature engineering for the demand prediction model.

All functions are pure transformations: DataFrame in, DataFrame out.
No DB calls, no side effects — easy to test and reuse in the API.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the demand_hourly table into a model-ready feature matrix.

    Input columns expected:
        zone_id, hour_start, trip_count, avg_distance, avg_fare,
        avg_passengers, borough (joined from zones)

    Returns a DataFrame with feature columns + target column 'trip_count'.
    The original index is preserved so you can align predictions back.
    """
    df = df.copy()

    # ── Time features ────────────────────────────────────────────────────────
    # Parse if not already datetime
    df["hour_start"] = pd.to_datetime(df["hour_start"])

    df["hour_of_day"] = df["hour_start"].dt.hour          # 0–23
    df["day_of_week"] = df["hour_start"].dt.dayofweek     # 0=Mon, 6=Sun
    df["month"] = df["hour_start"].dt.month                # 1–12
    df["day_of_month"] = df["hour_start"].dt.day
    df["week_of_year"] = df["hour_start"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["hour_start"].dt.quarter

    # Binary flags — these encode domain knowledge about NYC taxi patterns
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_late_night"] = df["hour_of_day"].isin([22, 23, 0, 1, 2, 3]).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # Cyclical encoding of hour and day — avoids the "gap" between 23 and 0
    # A linear encoding treats hour 23 and hour 0 as maximally different,
    # but they are adjacent. Sin/cos projects them onto a circle.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Zone features ─────────────────────────────────────────────────────────
    # zone_id as a categorical integer — XGBoost handles this natively
    df["zone_id"] = df["zone_id"].astype(int)

    # Borough one-hot (Manhattan, Brooklyn, Queens, Bronx, Staten Island + EWR)
    borough_dummies = pd.get_dummies(df["borough"], prefix="borough", dtype=int)
    df = pd.concat([df, borough_dummies], axis=1)

    # ── Lag features ──────────────────────────────────────────────────────────
    # "What was demand in this zone 1 hour ago? 24 hours ago? Last week?"
    # These are the single most predictive features in most demand models.
    # We sort within each zone before shifting to avoid cross-zone contamination.
    df = df.sort_values(["zone_id", "hour_start"])
    df["lag_1h"] = df.groupby("zone_id")["trip_count"].shift(1)
    df["lag_24h"] = df.groupby("zone_id")["trip_count"].shift(24)
    df["lag_168h"] = df.groupby("zone_id")["trip_count"].shift(168)  # 1 week

    # Rolling averages — smooth out noise, capture trend
    df["rolling_mean_24h"] = (
        df.groupby("zone_id")["trip_count"]
        .transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
    )
    df["rolling_mean_168h"] = (
        df.groupby("zone_id")["trip_count"]
        .transform(lambda x: x.shift(1).rolling(168, min_periods=1).mean())
    )

    # ── Trip characteristic features ─────────────────────────────────────────
    df["avg_distance"] = df["avg_distance"].fillna(0)
    df["avg_fare"] = df["avg_fare"].fillna(0)
    df["avg_passengers"] = df["avg_passengers"].fillna(1)

    return df


def get_feature_columns() -> list[str]:
    """
    Canonical ordered list of feature columns fed to the model.
    Must stay in sync with build_features() output.
    Order matters: SHAP values align to this list.
    """
    return [
        # Time raw
        "hour_of_day", "day_of_week", "month", "day_of_month",
        "week_of_year", "quarter",
        # Time binary
        "is_weekend", "is_rush_hour", "is_late_night", "is_monday", "is_friday",
        # Time cyclical
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Zone
        "zone_id",
        # Borough (present if data has all boroughs — handle missing gracefully)
        "borough_Bronx", "borough_Brooklyn", "borough_EWR",
        "borough_Manhattan", "borough_Queens", "borough_Staten Island",
        "borough_Unknown",
        # Lags
        "lag_1h", "lag_24h", "lag_168h",
        "rolling_mean_24h", "rolling_mean_168h",
        # Trip characteristics
        "avg_distance", "avg_fare", "avg_passengers",
    ]


TARGET = "trip_count"
