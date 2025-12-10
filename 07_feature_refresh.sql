-- ============================================================================
-- PHASE 7: Feature Refresh and Maintenance
-- Run this periodically to update features (daily/hourly recommended)
-- ============================================================================

USE WAREHOUSE fe_wh;
USE DATABASE fe_demo_db;
USE SCHEMA fe_features;

CREATE TABLE IF NOT EXISTS fe_features.feature_refresh_log (
    refresh_id STRING DEFAULT UUID_STRING(),
    refresh_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    records_updated NUMBER,
    status STRING,
    error_message STRING
);

DELETE FROM fe_features.customer_features;

INSERT INTO fe_features.customer_features
SELECT 
  *,
  CURRENT_TIMESTAMP() AS feature_timestamp
FROM fe_features.customer_features_v;

INSERT INTO fe_features.feature_refresh_log (records_updated, status)
SELECT 
  COUNT(*) AS records_updated,
  'SUCCESS' AS status
FROM fe_features.customer_features
WHERE feature_timestamp >= DATEADD('minute', -5, CURRENT_TIMESTAMP());

CREATE OR REPLACE TABLE fe_store.customer_features_online AS
SELECT *
FROM fe_features.customer_features
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY feature_timestamp DESC) = 1;

SELECT 'Features refreshed successfully!' AS status, CURRENT_TIMESTAMP() AS refresh_time;

SELECT 
    'Raw Events' AS data_type,
    COUNT(*) AS record_count,
    COUNT(DISTINCT customer_id) AS unique_customers,
    MIN(event_ts) AS earliest_event,
    MAX(event_ts) AS latest_event
FROM fe_raw.customer_events
UNION ALL
SELECT 
    'Engineered Features' AS data_type,
    COUNT(*) AS record_count,
    COUNT(DISTINCT customer_id) AS unique_customers,
    MIN(feature_timestamp) AS earliest_event,
    MAX(feature_timestamp) AS latest_event
FROM fe_features.customer_features
UNION ALL
SELECT 
    'Online Features' AS data_type,
    COUNT(*) AS record_count,
    COUNT(DISTINCT customer_id) AS unique_customers,
    MIN(feature_timestamp) AS earliest_event,
    MAX(feature_timestamp) AS latest_event
FROM fe_store.customer_features_online
UNION ALL
SELECT 
    'Training Labels' AS data_type,
    COUNT(*) AS record_count,
    COUNT(DISTINCT customer_id) AS unique_customers,
    NULL AS earliest_event,
    NULL AS latest_event
FROM fe_raw.labels;

SELECT TOP 10
    refresh_id,
    refresh_timestamp,
    records_updated,
    status,
    error_message
FROM fe_features.feature_refresh_log
ORDER BY refresh_timestamp DESC;

