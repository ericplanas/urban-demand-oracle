"""
load.py — Upsert clean trips and demand_hourly into PostgreSQL.

Uses COPY for bulk loading (fastest method — streams data via a buffer,
bypasses row-by-row INSERT overhead). For demand_hourly we use INSERT ON CONFLICT
to allow re-running the ETL without duplicating rows.
"""
import io
import logging
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import urllib.request
import csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [load] %(message)s")
log = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ.get("POSTGRES_PORT", 5432),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def load_trips(df: pd.DataFrame, conn) -> int:
    """Bulk-load trips using COPY FROM STDIN (fastest PostgreSQL bulk insert)."""
    cols = [
        "pickup_datetime", "dropoff_datetime",
        "pickup_zone_id", "dropoff_zone_id",
        "passenger_count", "trip_distance",
        "fare_amount", "tip_amount", "payment_type",
    ]
    df_subset = df[cols].copy()
    df_subset["payment_type"] = df_subset["payment_type"].fillna(0).astype(int)
    df_subset["passenger_count"] = df_subset["passenger_count"].fillna(1).astype(int)
    df_subset["tip_amount"] = df_subset["tip_amount"].fillna(0.0)

    buffer = io.StringIO()
    df_subset.to_csv(buffer, index=False, header=False)
    buffer.seek(0)

    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY trips ({', '.join(cols)}) FROM STDIN WITH CSV",
            buffer,
        )
    conn.commit()
    log.info(f"  Loaded {len(df_subset):,} rows into trips")
    return len(df_subset)


def upsert_demand_hourly(df: pd.DataFrame, conn) -> int:
    """
    Upsert demand_hourly. ON CONFLICT DO UPDATE means re-running ETL
    is safe — it recalculates aggregates rather than duplicating rows.
    """
    cols = [
        "zone_id", "hour_start", "trip_count",
        "avg_fare", "avg_distance",
        "hour_of_day", "day_of_week", "is_weekend", "month",
    ]
    rows = [tuple(row) for row in df[cols].itertuples(index=False)]

    sql = """
        INSERT INTO demand_hourly (
            zone_id, hour_start, trip_count,
            avg_fare, avg_distance,
            hour_of_day, day_of_week, is_weekend, month
        ) VALUES %s
        ON CONFLICT (zone_id, hour_start)
        DO UPDATE SET
            trip_count   = EXCLUDED.trip_count,
            avg_fare     = EXCLUDED.avg_fare,
            avg_distance = EXCLUDED.avg_distance,
            hour_of_day  = EXCLUDED.hour_of_day,
            day_of_week  = EXCLUDED.day_of_week,
            is_weekend   = EXCLUDED.is_weekend,
            month        = EXCLUDED.month
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    log.info(f"  Upserted {len(rows):,} rows into demand_hourly")
    return len(rows)


def load_zones(conn) -> int:
    """
    Download the official NYC TLC taxi zone lookup CSV and load into zones table.
    This is a static reference table — 265 rows, loaded once.
    Safe to re-run: truncates and reloads.
    """

    TLC_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

    log.info("Downloading NYC TLC zone lookup...")
    with urllib.request.urlopen(TLC_ZONES_URL) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    rows = [
        (int(r["LocationID"]), r["Borough"], r["Zone"], r["service_zone"])
        for r in reader
        if r["LocationID"].strip().isdigit()
    ]

    with conn.cursor() as cur:
        cur.execute("SET CONSTRAINTS fk_pickup_zone, fk_dropoff_zone DEFERRED")
        cur.execute("DELETE FROM zones")
        execute_values(
            cur,
            "INSERT INTO zones (zone_id, borough, zone_name, service_zone) VALUES %s",
            rows,
        )
    conn.commit()
    log.info(f"  Loaded {len(rows)} zones")
    return len(rows)


def load_weather(conn) -> int:
    """
    Fetch 2023 hourly weather for NYC from Open-Meteo (free, no API key).
    Variables: temperature_2m, precipitation, snowfall, windspeed_10m, weathercode.
    Safe to re-run: uses ON CONFLICT DO UPDATE.
    """
    import urllib.request
    import json

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        "?latitude=40.7829&longitude=-73.9654"
        "&start_date=2023-01-01&end_date=2023-12-31"
        "&hourly=temperature_2m,precipitation,snowfall,windspeed_10m,weathercode"
        "&timezone=America%2FNew_York"
    )

    log.info("Fetching 2023 hourly weather from Open-Meteo...")
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))

    hourly = data["hourly"]
    timestamps = hourly["time"]
    rows = [
        (
            ts,
            hourly["temperature_2m"][i],
            hourly["precipitation"][i],
            hourly["snowfall"][i],
            hourly["windspeed_10m"][i],
            hourly["weathercode"][i],
        )
        for i, ts in enumerate(timestamps)
    ]

    sql = """
        INSERT INTO weather
            (observed_at, temperature_c, precipitation_mm, snow_depth_mm, wind_speed_kmh, weather_code)
        VALUES %s
        ON CONFLICT (observed_at)
        DO UPDATE SET
            temperature_c     = EXCLUDED.temperature_c,
            precipitation_mm  = EXCLUDED.precipitation_mm,
            snow_depth_mm     = EXCLUDED.snow_depth_mm,
            wind_speed_kmh    = EXCLUDED.wind_speed_kmh,
            weather_code      = EXCLUDED.weather_code
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    log.info(f"  Loaded {len(rows)} weather rows")
    return len(rows)



if __name__ == "__main__":
    conn = get_connection()
    log.info("DB connection OK")
    conn.close()
