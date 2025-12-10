-- ============================================================================
-- PHASE 2: Feature Engineering
-- ============================================================================

USE WAREHOUSE fe_wh;
USE DATABASE fe_demo_db;
USE SCHEMA fe_features;

CREATE OR REPLACE VIEW fe_features.customer_features_v AS
WITH base AS (
  SELECT
    customer_id,
    event_ts,
    event_type,
    channel,
    amount
  FROM fe_raw.customer_events
),
agg AS (
  SELECT
    customer_id,
    MAX(event_ts) AS last_event_ts,
    DATEDIFF('day', MAX(event_ts), CURRENT_TIMESTAMP()) AS days_since_last_event,
    COUNT_IF(event_type='purchase') AS purchases_7d,
    SUM(CASE WHEN event_type='purchase' THEN amount ELSE 0 END) AS revenue_7d,
    COUNT_IF(event_type='refund') AS refunds_7d,
    SUM(CASE WHEN event_type='refund' THEN amount ELSE 0 END) AS refund_amount_7d,
    COUNT(*) AS events_7d,
    COUNT_IF(channel='mobile') AS mobile_events_7d,
    COUNT_IF(channel='web') AS web_events_7d,
    AVG(CASE WHEN amount<>0 THEN amount END) AS avg_amount_nonzero_7d,
    COUNT_IF(event_type='browse') AS browse_events_7d,
    STDDEV(amount) AS amount_stddev_7d,
    MAX(amount) AS max_amount_7d,
    MIN(amount) AS min_amount_7d
  FROM base
  GROUP BY customer_id
)
SELECT * FROM agg;

CREATE OR REPLACE TABLE fe_features.customer_features AS
SELECT 
  *,
  CURRENT_TIMESTAMP() AS feature_timestamp
FROM fe_features.customer_features_v;

SELECT * FROM fe_features.customer_features ORDER BY customer_id;

CREATE OR REPLACE VIEW fe_features.customer_features_online AS
SELECT *
FROM fe_features.customer_features
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY feature_timestamp DESC) = 1;

