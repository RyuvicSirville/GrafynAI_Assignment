-- ============================================================================
-- PHASE 4: Create Training Labels and Offline Feature Retrieval
-- ============================================================================

USE WAREHOUSE fe_wh;
USE DATABASE fe_demo_db;
USE SCHEMA fe_raw;

DELETE FROM fe_features.customer_features;
INSERT INTO fe_features.customer_features
SELECT *, CURRENT_TIMESTAMP() AS feature_timestamp
FROM fe_features.customer_features_v;

CREATE OR REPLACE VIEW fe_features.customer_features_online AS
SELECT *
FROM fe_features.customer_features
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY feature_timestamp DESC) = 1;

CREATE OR REPLACE TABLE fe_raw.labels (
  customer_id STRING,
  label_ts    TIMESTAMP_NTZ,
  churned     BOOLEAN
);

INSERT INTO fe_raw.labels VALUES
  ('C001', CURRENT_TIMESTAMP(), FALSE),
  ('C002', CURRENT_TIMESTAMP(), FALSE),
  ('C003', CURRENT_TIMESTAMP(), TRUE),
  ('C004', CURRENT_TIMESTAMP(), FALSE),
  ('C005', CURRENT_TIMESTAMP(), TRUE),
  ('C006', CURRENT_TIMESTAMP(), FALSE),
  ('C007', CURRENT_TIMESTAMP(), TRUE),
  ('C008', CURRENT_TIMESTAMP(), FALSE),
  ('C009', CURRENT_TIMESTAMP(), FALSE),
  ('C010', CURRENT_TIMESTAMP(), TRUE),
  ('C011', CURRENT_TIMESTAMP(), FALSE),
  ('C012', CURRENT_TIMESTAMP(), FALSE),
  ('C013', CURRENT_TIMESTAMP(), TRUE),
  ('C014', CURRENT_TIMESTAMP(), FALSE),
  ('C015', CURRENT_TIMESTAMP(), TRUE),
  ('C016', CURRENT_TIMESTAMP(), FALSE),
  ('C017', CURRENT_TIMESTAMP(), TRUE),
  ('C018', CURRENT_TIMESTAMP(), FALSE),
  ('C019', CURRENT_TIMESTAMP(), FALSE),
  ('C020', CURRENT_TIMESTAMP(), TRUE),
  ('C021', CURRENT_TIMESTAMP(), FALSE),
  ('C022', CURRENT_TIMESTAMP(), TRUE),
  ('C023', CURRENT_TIMESTAMP(), FALSE),
  ('C024', CURRENT_TIMESTAMP(), TRUE),
  ('C025', CURRENT_TIMESTAMP(), FALSE);

SELECT * FROM fe_raw.labels;

SELECT 
    'Features Check' AS check_type,
    COUNT(*) AS feature_count,
    MIN(feature_timestamp) AS min_feature_ts,
    MAX(feature_timestamp) AS max_feature_ts
FROM fe_features.customer_features;

SELECT 
    'Labels Check' AS check_type,
    COUNT(*) AS label_count,
    MIN(label_ts) AS min_label_ts,
    MAX(label_ts) AS max_label_ts
FROM fe_raw.labels;

CREATE OR REPLACE VIEW fe_features.training_set AS
WITH ranked AS (
  SELECT 
    f.*,
    l.label_ts,
    l.churned,
    ROW_NUMBER() OVER (
      PARTITION BY l.customer_id, l.label_ts
      ORDER BY f.feature_timestamp DESC
    ) AS rn
  FROM fe_features.customer_features f
  JOIN fe_raw.labels l
    ON f.customer_id = l.customer_id
   AND f.feature_timestamp <= l.label_ts
)
SELECT 
  customer_id,
  days_since_last_event,
  purchases_7d,
  revenue_7d,
  refunds_7d,
  refund_amount_7d,
  events_7d,
  mobile_events_7d,
  web_events_7d,
  avg_amount_nonzero_7d,
  browse_events_7d,
  amount_stddev_7d,
  max_amount_7d,
  min_amount_7d,
  churned AS label
FROM ranked
WHERE rn = 1;

SELECT * FROM fe_features.training_set ORDER BY customer_id;

