"""
ML training pipeline for Urban Demand Oracle.

Run order:
    1. Load demand_hourly + zones from Postgres
    2. Feature engineering (features.py)
    3. Chronological train/test split (no data leakage)
    4. XGBoost training with early stopping
    5. SHAP value computation
    6. Persist model + SHAP values to data/processed/
    7. Write model metadata to models table in Postgres
"""

import os
import pickle
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
import shap
from sqlalchemy import create_engine, text

from features import build_features, get_feature_columns, TARGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_URL = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
MODEL_DIR = Path("/app/data/processed")
MODEL_PATH = MODEL_DIR / "model.pkl"
SHAP_PATH = MODEL_DIR / "shap_values.npy"
META_PATH = MODEL_DIR / "model_meta.json"

# XGBoost hyperparameters — reasonable defaults for a demand forecasting task
# These are not tuned; consider Optuna for a next iteration
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,   # prevents overfitting on rare zone-hour combos
    "reg_alpha": 0.1,         # L1 — encourages sparse feature use
    "reg_lambda": 1.0,        # L2 — shrinks leaf weights
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}

# Train on first 10 months, validate on last 2 (Nov–Dec 2023)
# This is a strict temporal split — no future data leaks into training
SPLIT_DATE = "2023-11-01"


def load_data(engine) -> pd.DataFrame:
    log.info("Loading demand_hourly + zones from Postgres …")
    query = """
        SELECT
            d.zone_id,
            d.hour_start,
            d.trip_count,
            d.avg_distance,
            d.avg_fare,
            d.avg_passengers,
            z.borough
        FROM demand_hourly d
        JOIN zones z ON d.zone_id = z.zone_id
        ORDER BY d.zone_id, d.hour_start
    """
    df = pd.read_sql(query, engine)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def make_feature_matrix(df: pd.DataFrame):
    """
    Build features, drop rows with NaN lags (first 168h per zone),
    return X (feature matrix) and y (target), plus the split index.
    """
    log.info("Engineering features …")
    df = build_features(df)

    feature_cols = get_feature_columns()
    # Only keep columns that actually exist (borough dummies depend on data)
    available = [c for c in feature_cols if c in df.columns]
    missing_boros = [c for c in feature_cols if c not in df.columns]
    if missing_boros:
        log.warning(f"Missing borough columns (filling 0): {missing_boros}")
        for col in missing_boros:
            df[col] = 0

    # Drop rows where lag features are NaN (first week per zone has no history)
    lag_cols = ["lag_1h", "lag_24h", "lag_168h"]
    before = len(df)
    df = df.dropna(subset=lag_cols)
    log.info(f"Dropped {before - len(df):,} rows with NaN lags (expected: first 168h per zone)")

    X = df[feature_cols]
    y = df[TARGET]
    dates = df["hour_start"]

    return X, y, dates


def chronological_split(X, y, dates):
    """
    Split at SPLIT_DATE. This is the only correct way to split time series data.
    A random split would let the model 'see' future trips, inflating metrics.
    """
    mask = dates < SPLIT_DATE
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]
    log.info(
        f"Train: {len(X_train):,} rows (up to {SPLIT_DATE}) | "
        f"Test: {len(X_test):,} rows"
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test) -> xgb.XGBRegressor:
    log.info("Training XGBoost …")
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    best = model.best_iteration
    log.info(f"Best iteration: {best}")
    return model


def compute_shap(model, X_train) -> np.ndarray:
    """
    Compute SHAP values on a 2000-row sample of training data.
    Full computation on millions of rows is slow; a sample is representative.
    TreeExplainer is exact for tree models (not approximate).
    """
    log.info("Computing SHAP values (sample of 2000 rows) …")
    sample = X_train.sample(min(2000, len(X_train)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    log.info(f"SHAP matrix shape: {shap_values.shape}")
    return shap_values


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)  # demand can't be negative

    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    # MAPE only on non-zero actuals to avoid division by zero
    mask = y_test > 0
    mape = float(np.mean(np.abs((preds[mask] - y_test[mask]) / y_test[mask])) * 100)

    log.info(f"MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_artifacts(model, shap_values, metrics, feature_cols):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {MODEL_PATH}")

    np.save(SHAP_PATH, shap_values)
    log.info(f"SHAP values saved → {SHAP_PATH}")

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "split_date": SPLIT_DATE,
        "feature_columns": feature_cols,
        "metrics": metrics,
        "xgboost_params": {k: v for k, v in XGBOOST_PARAMS.items()},
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved → {META_PATH}")

    return meta


def write_model_record(engine, metrics, meta):
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO models (version, trained_at, rmse, mae, notes, artifact_path)
                VALUES (:version, :trained_at, :rmse, :mae, :notes, :artifact_path)
            """),
            {
                "version": version,
                "trained_at": meta["trained_at"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "notes": json.dumps({"mape": metrics["mape"], **meta["xgboost_params"]}),
                "artifact_path": str(MODEL_PATH),
            },
        )
    log.info(f"Model record written to DB: {version}")


def main():
    engine = create_engine(DB_URL)

    df = load_data(engine)
    X, y, dates = make_feature_matrix(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y, dates)

    model = train_model(X_train, X_test, y_train, y_test)
    shap_values = compute_shap(model, X_train)
    metrics = evaluate(model, X_test, y_test)
    meta = save_artifacts(model, shap_values, metrics, list(X.columns))

    try:
        write_model_record(engine, metrics, meta)
    except Exception as e:
        log.warning(f"Could not write model record to DB (non-fatal): {e}")

    log.info("Training complete ✓")


if __name__ == "__main__":
    main()
