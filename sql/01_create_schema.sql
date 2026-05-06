-- AutoRec Star Schema
-- Run once on a fresh database: psql -d autorec -f sql/01_create_schema.sql

-- ============================================================
-- Dimension: User
-- ============================================================
CREATE TABLE IF NOT EXISTS dim_user (
    user_id          VARCHAR(10)  PRIMARY KEY,
    age              SMALLINT     NOT NULL CHECK (age BETWEEN 18 AND 75),
    gender           CHAR(1)      NOT NULL CHECK (gender IN ('M', 'F')),
    income_bracket   VARCHAR(4)   NOT NULL CHECK (income_bracket IN ('low', 'mid', 'high')),
    city             VARCHAR(100) NOT NULL,
    registration_date DATE        NOT NULL
);

-- ============================================================
-- Dimension: Car
-- ============================================================
CREATE TABLE IF NOT EXISTS dim_car (
    car_id           VARCHAR(10)  PRIMARY KEY,
    make             VARCHAR(50)  NOT NULL,
    model            VARCHAR(100) NOT NULL,
    year             SMALLINT     NOT NULL CHECK (year BETWEEN 1990 AND 2030),
    body_type        VARCHAR(20)  NOT NULL,
    fuel_type        VARCHAR(20)  NOT NULL,
    transmission     VARCHAR(20)  NOT NULL,
    price            INTEGER      NOT NULL CHECK (price BETWEEN 1000 AND 500000),
    price_tier       VARCHAR(10)  NOT NULL CHECK (price_tier IN ('budget', 'mid', 'premium', 'luxury')),
    mileage          INTEGER      NOT NULL CHECK (mileage >= 0),
    age_at_listing   SMALLINT     NOT NULL CHECK (age_at_listing >= 0)
);

-- ============================================================
-- Dimension: Date  (pre-populated calendar table)
-- ============================================================
CREATE TABLE IF NOT EXISTS dim_date (
    date_id     INTEGER PRIMARY KEY,   -- surrogate key: YYYYMMDD
    full_date   DATE    NOT NULL UNIQUE,
    year        SMALLINT NOT NULL,
    quarter     SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month       SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    day_of_week SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),  -- 0=Sunday
    is_weekend  BOOLEAN  NOT NULL
);

-- Populate dim_date for the range covered by our data (2022-01-01 to 2024-12-31)
INSERT INTO dim_date (date_id, full_date, year, quarter, month, day_of_week, is_weekend)
SELECT
    TO_CHAR(d, 'YYYYMMDD')::INTEGER                         AS date_id,
    d                                                        AS full_date,
    EXTRACT(YEAR    FROM d)::SMALLINT                        AS year,
    EXTRACT(QUARTER FROM d)::SMALLINT                        AS quarter,
    EXTRACT(MONTH   FROM d)::SMALLINT                        AS month,
    EXTRACT(DOW     FROM d)::SMALLINT                        AS day_of_week,
    EXTRACT(DOW     FROM d) IN (0, 6)                        AS is_weekend
FROM GENERATE_SERIES(
    '2022-01-01'::DATE,
    '2024-12-31'::DATE,
    '1 day'::INTERVAL
) AS d
ON CONFLICT (date_id) DO NOTHING;

-- ============================================================
-- Fact: Interactions
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_interactions (
    interaction_id        VARCHAR(10) PRIMARY KEY,
    user_id               VARCHAR(10) NOT NULL REFERENCES dim_user(user_id),
    car_id                VARCHAR(10) NOT NULL REFERENCES dim_car(car_id),
    date_id               INTEGER     NOT NULL REFERENCES dim_date(date_id),
    interaction_type      VARCHAR(15) NOT NULL CHECK (interaction_type IN ('test_drive', 'purchase')),
    interaction_timestamp TIMESTAMP   NOT NULL
);
