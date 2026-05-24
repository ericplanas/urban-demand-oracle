"""
ingest.py — Download NYC Yellow Taxi 2023 Parquet files directly from the TLC.
Saves raw files to /app/data/raw/. Skips files already downloaded.
No API key required — TLC data is publicly accessible.
"""
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ingest] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path("/app/data/raw")
TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def ingest():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for month in MONTHS:
        filename = f"yellow_tripdata_2023-{month}.parquet"
        dest = RAW_DIR / filename
        if dest.exists():
            log.info(f"  {filename} already present, skipping")
            continue
        url = f"{TLC_BASE}/{filename}"
        log.info(f"  Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        log.info(f"  {filename} saved ({dest.stat().st_size / 1_048_576:.1f} MB)")

    files = list(RAW_DIR.glob("yellow_tripdata_2023-*.parquet"))
    log.info(f"Found {len(files)} Parquet files")
    if not files:
        raise FileNotFoundError("No 2023 Parquet files found in data/raw")
    return sorted(files)


if __name__ == "__main__":
    ingest()
