"""
main.py — ETL entrypoint. Orchestrates: ingest → clean → transform → load.
Reference/weather tables are loaded first (fast, static or one-year fetch).
Trip files are processed one by one to keep memory footprint small.
"""
import logging
from ingest import ingest
from clean import clean_file
from transform import build_demand_hourly
from load import get_connection, load_trips, upsert_demand_hourly, load_zones, load_weather

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ETL] %(message)s")
log = logging.getLogger(__name__)


def run():
    log.info("=== ETL pipeline starting ===")

    conn = get_connection()

    # --- Reference tables first (fast, idempotent) ---
    load_zones(conn)

    # --- Trip files ---
    parquet_files = ingest()
    parquet_files = sorted(parquet_files)
    log.info(f"Processing {len(parquet_files)} Parquet files")

    total_trips = 0
    total_demand_rows = 0

    for path in parquet_files:
        log.info(f"--- Processing {path.name} ---")

        df_clean = clean_file(path)
        if df_clean.empty:
            log.warning(f"  No valid rows in {path.name}, skipping")
            continue

        df_demand = build_demand_hourly(df_clean)

        total_trips += load_trips(df_clean, conn)
        total_demand_rows += upsert_demand_hourly(df_demand, conn)

    # --- Weather loaded last — independent of trips, failure is non-fatal ---
    try:
        load_weather(conn)
    except Exception as e:
        log.warning(f"Weather load failed (non-fatal): {e}")

    conn.close()
    log.info(f"=== ETL complete: {total_trips:,} trips | {total_demand_rows:,} demand rows ===")


if __name__ == "__main__":
    run()
