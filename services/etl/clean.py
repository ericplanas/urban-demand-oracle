"""
clean.py — Load a raw Parquet file, apply quality filters, return a clean DataFrame.

Quality rules (based on NYC TLC data dictionary + common sense):
- Drop rows with null pickup/dropoff datetime or zone IDs
- Keep trips between 1 Jan 2023 and 31 Dec 2023
- Trip distance: 0.1 – 100 miles
- Fare amount: $2.50 – $500
- Passenger count: 1 – 6
- Zone IDs must be in the valid NYC TLC range (1–265)
"""
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [clean] %(message)s")
log = logging.getLogger(__name__)

YEAR = 2023
REQUIRED_COLS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID",
    "passenger_count", "trip_distance",
    "fare_amount", "tip_amount", "payment_type",
]


def clean_file(path: Path) -> pd.DataFrame:
    log.info(f"Cleaning {path.name}")
    df = pd.read_parquet(path, columns=REQUIRED_COLS)
    n_raw = len(df)

    # Rename to our schema names
    df = df.rename(columns={
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "PULocationID": "pickup_zone_id",
        "DOLocationID": "dropoff_zone_id",
    })

    # Drop nulls in critical columns
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime", "pickup_zone_id", "dropoff_zone_id"])

    # Type coercion
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime"])

    # Time range filter — keep only 2023
    df = df[(df["pickup_datetime"].dt.year == YEAR)]

    # Numeric filters
    df = df[df["trip_distance"].between(0.1, 100)]
    df = df[df["fare_amount"].between(2.50, 500)]
    df = df[df["passenger_count"].between(1, 6)]
    df = df[df["pickup_zone_id"].between(1, 265)]
    df = df[df["dropoff_zone_id"].between(1, 265)]

    # Cast zone IDs to int
    df["pickup_zone_id"] = df["pickup_zone_id"].astype(int)
    df["dropoff_zone_id"] = df["dropoff_zone_id"].astype(int)

    n_clean = len(df)
    log.info(f"  {path.name}: {n_raw:,} raw → {n_clean:,} clean ({n_raw - n_clean:,} dropped)")
    return df


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1])
    df = clean_file(path)
    print(df.head())
