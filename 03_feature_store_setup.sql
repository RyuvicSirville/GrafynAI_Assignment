-- ============================================================================
-- PHASE 3: Feature Store Setup (SQL-Only Pattern)
-- Use this if Snowpark ML Feature Store is not available
-- ============================================================================

USE WAREHOUSE fe_wh;
USE DATABASE fe_demo_db;
USE SCHEMA fe_store;

CREATE OR REPLACE TABLE fe_store.entity_registry (
    entity_name STRING,
    identifier_column STRING,
    description STRING,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO fe_store.entity_registry (entity_name, identifier_column, description)
VALUES ('customer_entity', 'customer_id', 'Customer entity for feature lookup');

CREATE OR REPLACE TABLE fe_store.feature_view_registry (
    feature_view_name STRING,
    entity_name STRING,
    source_table STRING,
    feature_columns STRING,
    description STRING,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO fe_store.feature_view_registry (
    feature_view_name, entity_name, source_table, feature_columns, description
)
VALUES (
    'customer_features_fv',
    'customer_entity',
    'fe_features.customer_features',
    'customer_id,last_event_ts,days_since_last_event,purchases_7d,revenue_7d,refunds_7d,refund_amount_7d,events_7d,mobile_events_7d,web_events_7d,avg_amount_nonzero_7d,browse_events_7d,amount_stddev_7d,max_amount_7d,min_amount_7d,feature_timestamp',
    'Feature view for customer aggregated features'
);

CREATE OR REPLACE VIEW fe_store.customer_features_fv AS
SELECT
    customer_id,
    last_event_ts,
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
    feature_timestamp
FROM fe_features.customer_features;

CREATE OR REPLACE TABLE fe_store.customer_features_online AS
SELECT * FROM fe_features.customer_features_online;

SELECT 'Entity Registry' AS registry_type, COUNT(*) AS count FROM fe_store.entity_registry
UNION ALL
SELECT 'Feature View Registry' AS registry_type, COUNT(*) AS count FROM fe_store.feature_view_registry
UNION ALL
SELECT 'Feature Records' AS registry_type, COUNT(*) AS count FROM fe_store.customer_features_online;

SELECT * FROM fe_store.entity_registry;

SELECT * FROM fe_store.feature_view_registry;

