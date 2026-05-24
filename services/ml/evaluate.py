"""
Standalone evaluation script — load a saved model and score it on the test set.
Useful for re-evaluating after retraining without touching train.py.
"""

import os
import pickle
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from features import build_features, get_feature_columns, TARGET
from train import load_data, make_feature_matrix, chronological_split, evaluate, SPLIT_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_URL = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)
MODEL_PATH = Path("/app/data/processed/model.pkl")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run train.py first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    log.info(f"Loaded model from {MODEL_PATH}")

    engine = create_engine(DB_URL)
    df = load_data(engine)
    X, y, dates = make_feature_matrix(df)
    _, X_test, _, y_test = chronological_split(X, y, dates)

    metrics = evaluate(model, X_test, y_test)

    print("\n── Evaluation Results ──────────────────────")
    print(f"  MAE  : {metrics['mae']:.2f} trips")
    print(f"  RMSE : {metrics['rmse']:.2f} trips")
    print(f"  MAPE : {metrics['mape']:.2f}%")
    print(f"  Test set: {SPLIT_DATE} → end of dataset")
    print("────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()