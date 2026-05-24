-- ============================================================
-- Urban Demand Oracle — PostgreSQL Schema
-- ============================================================

-- ------------------------------------------------------------
-- 1. ZONES — NYC Taxi Zone lookup table (265 zones)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS zones (
    zone_id       SMALLINT    PRIMARY KEY,
    borough       VARCHAR(32) NOT NULL,
    zone_name     VARCHAR(64) NOT NULL,
    service_zone  VARCHAR(32)               -- e.g. 'Yellow Zone', 'Boro Zone', 'Airports'
);

-- ------------------------------------------------------------
-- 2. TRIPS — Core fact table (one row per taxi trip)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trips (
    trip_id            BIGSERIAL    PRIMARY KEY,
    pickup_datetime    TIMESTAMP    NOT NULL,
    dropoff_datetime   TIMESTAMP    NOT NULL,
    pickup_zone_id     SMALLINT,
    dropoff_zone_id    SMALLINT,
    passenger_count    SMALLINT,
    trip_distance      NUMERIC(8,2),
    fare_amount        NUMERIC(8,2),
    tip_amount         NUMERIC(8,2),
    total_amount       NUMERIC(8,2),
    payment_type       SMALLINT,    -- 1=Credit, 2=Cash, 3=No charge, 4=Dispute
    store_and_fwd_flag CHAR(1),
    CONSTRAINT fk_pickup_zone  FOREIGN KEY (pickup_zone_id)  REFERENCES zones(zone_id) DEFERRABLE INITIALLY IMMEDIATE,
    CONSTRAINT fk_dropoff_zone FOREIGN KEY (dropoff_zone_id) REFERENCES zones(zone_id) DEFERRABLE INITIALLY IMMEDIATE
);

-- Indexes for the most common query patterns
CREATE INDEX IF NOT EXISTS idx_trips_pickup_datetime  ON trips (pickup_datetime);
CREATE INDEX IF NOT EXISTS idx_trips_pickup_zone_id   ON trips (pickup_zone_id);
CREATE INDEX IF NOT EXISTS idx_trips_dropoff_zone_id  ON trips (dropoff_zone_id);
-- Composite index: zone + time (used in demand aggregation queries)
CREATE INDEX IF NOT EXISTS idx_trips_zone_time        ON trips (pickup_zone_id, pickup_datetime);

-- ------------------------------------------------------------
-- 3. WEATHER — Hourly weather observations (Central Park station)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather (
    weather_id        SERIAL       PRIMARY KEY,
    observed_at       TIMESTAMP    NOT NULL UNIQUE,  -- truncated to hour
    temperature_c     NUMERIC(5,2),
    precipitation_mm  NUMERIC(6,2),
    wind_speed_kmh    NUMERIC(5,2),
    snow_depth_mm     NUMERIC(6,2),
    weather_code      SMALLINT     -- WMO weather interpretation code
);

CREATE INDEX IF NOT EXISTS idx_weather_observed_at ON weather (observed_at);

-- ------------------------------------------------------------
-- 4. DEMAND_HOURLY — Pre-aggregated demand (ML training target)
-- ------------------------------------------------------------
-- One row = one zone × one hour
-- Populated by the ETL pipeline, read by the ML training service
CREATE TABLE IF NOT EXISTS demand_hourly (
    demand_id         BIGSERIAL    PRIMARY KEY,
    zone_id           SMALLINT     NOT NULL REFERENCES zones(zone_id),
    hour_start        TIMESTAMP    NOT NULL,  -- e.g. 2023-01-15 08:00:00
    trip_count        INTEGER      NOT NULL DEFAULT 0,
    avg_fare          NUMERIC(8,2),
    avg_distance      NUMERIC(8,2),
    -- Denormalised time features for fast ML feature access at training time
    hour_of_day       SMALLINT,
    day_of_week       SMALLINT,
    is_weekend        SMALLINT,
    month             SMALLINT,
    UNIQUE (zone_id, hour_start)
);

CREATE INDEX IF NOT EXISTS idx_demand_hourly_hour_start ON demand_hourly (hour_start);
CREATE INDEX IF NOT EXISTS idx_demand_hourly_zone_id    ON demand_hourly (zone_id);
-- Composite: the ML feature query pattern
CREATE INDEX IF NOT EXISTS idx_demand_zone_hour         ON demand_hourly (zone_id, hour_start);

-- ------------------------------------------------------------
-- 5. MODELS — Trained model registry
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
    model_id      SERIAL       PRIMARY KEY,
    version       VARCHAR(32)  NOT NULL UNIQUE,  -- e.g. 'v1.0.0', 'v1.1.0'
    trained_at    TIMESTAMP    NOT NULL DEFAULT NOW(),
    algorithm     VARCHAR(32)  NOT NULL DEFAULT 'xgboost',
    rmse          NUMERIC(10,4),
    mae           NUMERIC(10,4),
    r2_score      NUMERIC(8,6),
    n_train_rows  INTEGER,
    artifact_path VARCHAR(256),  -- path to .pkl / .joblib file inside the container
    is_active     BOOLEAN      NOT NULL DEFAULT FALSE,
    notes         TEXT
);

-- Only one model should be active at a time
-- Enforced at application level; this partial index makes it fast to find
CREATE UNIQUE INDEX IF NOT EXISTS idx_models_active
    ON models (is_active) WHERE is_active = TRUE;

-- ============================================================
-- Schema created successfully
-- ============================================================
