-- AutoRec Index Definitions
-- Run after 01_create_schema.sql and after data has been loaded.

-- 1. B-tree on user_id
--    Accelerates all per-user lookups: "show me this user's history",
--    collaborative-filter candidate retrieval, and JOIN to dim_user.
CREATE INDEX IF NOT EXISTS idx_fact_user_id
    ON fact_interactions (user_id);

-- 2. B-tree on car_id
--    Accelerates item-based lookups: "which users interacted with this car?",
--    item-similarity computation, and JOIN to dim_car.
CREATE INDEX IF NOT EXISTS idx_fact_car_id
    ON fact_interactions (car_id);

-- 3. Composite B-tree on (user_id, interaction_timestamp DESC)
--    Serves "get the N most recent interactions for a user" in a single
--    index scan — avoids a sort step on the timestamp column.
CREATE INDEX IF NOT EXISTS idx_fact_user_ts
    ON fact_interactions (user_id, interaction_timestamp DESC);

-- 4. Partial index on purchase rows only
--    Purchase events are a small fraction of interactions but are queried
--    heavily for conversion-rate analysis and purchase-based CF.
--    Keeping the index small improves cache efficiency.
CREATE INDEX IF NOT EXISTS idx_fact_purchases
    ON fact_interactions (user_id, car_id)
    WHERE interaction_type = 'purchase';

-- 5. Composite index on dim_car(price_tier, body_type)
--    Used by the recommendation API to filter the candidate car pool
--    (e.g. "find all mid-tier SUVs") before scoring, avoiding a full
--    table scan on dim_car for every online request.
CREATE INDEX IF NOT EXISTS idx_car_tier_body
    ON dim_car (price_tier, body_type);
