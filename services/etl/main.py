"""
main.py — ETL entrypoint. Orchestrates: ingest → clean → transform → load.

Processes files one by one to keep memory footprint small.
On a 16GB machine, each monthly Parquet file fits comfortably after filtering.
"""
import logging
from ingest import ingest
from clean import clean_file
from transform import build_demand_hourly
from load import get_connection, load_trips, upsert_demand_hourly

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ETL] %(message)s")
log = logging.getLogger(__name__)


def run():
    log.info("=== ETL pipeline starting ===")

    # Step 1 — Download raw files
    parquet_files = ingest()
    parquet_files = sorted(parquet_files)  # process chronologically
    log.info(f"Processing {len(parquet_files)} files")

    conn = get_connection()

    total_trips = 0
    total_demand_rows = 0

    for path in parquet_files:
        log.info(f"--- Processing {path.name} ---")

        # Step 2 — Clean
        df_clean = clean_file(path)
        if df_clean.empty:
            log.warning(f"  No valid rows in {path.name}, skipping")
            continue

        # Step 3 — Transform
        df_demand = build_demand_hourly(df_clean)

        # Step 4 — Load
        total_trips += load_trips(df_clean, conn)
        total_demand_rows += upsert_demand_hourly(df_demand, conn)

    conn.close()
    log.info(f"=== ETL complete: {total_trips:,} trips, {total_demand_rows:,} demand rows ===")


if __name__ == "__main__":
    run()
