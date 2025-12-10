# ============================================================================
# PHASE 6: Online Feature Retrieval for Inference
# Run this in a Snowflake Python Worksheet
# ============================================================================

import pandas as pd
import pickle
import tempfile
import os
import json

try:
    _ = session
    print("✅ Using session from notebook context")
except NameError:
    try:
        import snowflake.snowpark.context as snowpark_context
        session = snowpark_context.get_active_session()
        print("✅ Got session from snowpark context")
    except Exception as e:
        raise RuntimeError(
            f"Could not obtain Snowflake session. Are you running in a Snowflake notebook? ({e})"
        )

session.use_warehouse("FE_WH")
session.use_database("FE_DEMO_DB")
session.use_schema("FE_FEATURES")

feature_columns = [
    'days_since_last_event',
    'purchases_7d',
    'revenue_7d',
    'refunds_7d',
    'refund_amount_7d',
    'events_7d',
    'mobile_events_7d',
    'web_events_7d',
    'avg_amount_nonzero_7d',
    'browse_events_7d',
    'amount_stddev_7d',
    'max_amount_7d',
    'min_amount_7d'
]

def get_customer_features(customer_id: str):
    """
    Retrieve latest features for a customer (online inference)
    """
    features_df = session.sql(f"""
        SELECT *
        FROM fe_store.customer_features_online
        WHERE customer_id = '{customer_id}'
    """).to_pandas()
    
    features_df.columns = [c.lower() for c in features_df.columns]
    
    return features_df

def load_model_from_stage(model_name='rf_model.pkl'):
    """Load the saved model from Snowflake stage"""
    stage_name = "fe_features.model_stage"
    
    try:
        temp_dir = tempfile.gettempdir()
        
        session.file.get(
            stage_location=f"@{stage_name}/{model_name}",
            target_directory=temp_dir
        )
        
        local_path = os.path.join(temp_dir, model_name)
        
        with open(local_path, 'rb') as f:
            model = pickle.load(f)
        
        os.unlink(local_path)
        return model
    except Exception as e:
        print(f"⚠️  Could not load {model_name} from stage: {e}")
        return None

def predict_churn_batch(customer_ids: list, model=None):
    """
    Predict churn probability for multiple customers
    """
    if model is None:
        model = load_model_from_stage('rf_model.pkl')
        if model is None:
            print("⚠️  Model not found in stage. Loading from training set...")
            from sklearn.ensemble import RandomForestClassifier
            training_df = session.sql("SELECT * FROM fe_features.training_set").to_pandas()
            training_df.columns = [c.lower() for c in training_df.columns]
            X_train = training_df[feature_columns].fillna(0)
            y_train = training_df['label'].astype(int)
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            model.fit(X_train, y_train)
    
    results = []
    
    for customer_id in customer_ids:
        features_df = get_customer_features(customer_id)
        
        if features_df.empty:
            results.append({
                'customer_id': customer_id,
                'status': 'NOT_FOUND',
                'churn_probability': None,
                'churn_prediction': None
            })
            continue
        
        try:
            X_inference = features_df[feature_columns].fillna(0)
            
            churn_probability = model.predict_proba(X_inference)[0, 1]
            churn_prediction = model.predict(X_inference)[0]
            
            results.append({
                'customer_id': customer_id,
                'status': 'SUCCESS',
                'churn_probability': float(churn_probability),
                'churn_prediction': bool(churn_prediction),
                'key_features': {
                    'purchases_7d': float(features_df['purchases_7d'].iloc[0]),
                    'revenue_7d': float(features_df['revenue_7d'].iloc[0]),
                    'days_since_last_event': float(features_df['days_since_last_event'].iloc[0]),
                    'events_7d': float(features_df['events_7d'].iloc[0])
                }
            })
        except Exception as e:
            results.append({
                'customer_id': customer_id,
                'status': 'ERROR',
                'error': str(e)
            })
    
    return results

print("="*60)
print("ONLINE FEATURE RETRIEVAL AND INFERENCE")
print("="*60)

print("\n--- Loading all customers for batch prediction ---")
all_customers_df = session.sql("""
    SELECT DISTINCT customer_id 
    FROM fe_features.training_set 
    ORDER BY customer_id
""").to_pandas()

all_customers_df.columns = [c.lower() for c in all_customers_df.columns]

all_customer_ids = all_customers_df['customer_id'].tolist()
print(f"Found {len(all_customer_ids)} customers for prediction")

print("\n--- Running Batch Churn Prediction ---")
predictions = predict_churn_batch(all_customer_ids)

print("\n📊 CHURN PREDICTION RESULTS:")
print("-" * 80)

churn_results_df = pd.DataFrame([
    {
        'Customer ID': p['customer_id'],
        'Status': p['status'],
        'Churn Probability': f"{p['churn_probability']:.4f}" if p.get('churn_probability') else 'N/A',
        'Prediction': 'CHURN RISK' if p.get('churn_prediction') else 'RETAIN',
        'Purchases (7d)': f"{p['key_features']['purchases_7d']:.0f}" if p.get('key_features') else 'N/A',
        'Revenue (7d)': f"${p['key_features']['revenue_7d']:.2f}" if p.get('key_features') else 'N/A'
    }
    for p in predictions if p['status'] == 'SUCCESS'
])

print(churn_results_df.to_string(index=False))

print("\n--- Saving predictions to Snowflake ---")
try:
    predictions_for_sf = []
    for p in predictions:
        if p['status'] == 'SUCCESS':
            predictions_for_sf.append({
                'CUSTOMER_ID': p['customer_id'],
                'CHURN_PROBABILITY': p['churn_probability'],
                'CHURN_PREDICTION': p['churn_prediction'],
                'PREDICTION_TIMESTAMP': pd.Timestamp.now(),
                'PURCHASES_7D': p['key_features']['purchases_7d'],
                'REVENUE_7D': p['key_features']['revenue_7d']
            })
    
    if predictions_for_sf:
        predictions_pd = pd.DataFrame(predictions_for_sf)
        
        session.sql("CREATE OR REPLACE TABLE fe_features.inference_results (customer_id STRING, churn_probability FLOAT, churn_prediction BOOLEAN, prediction_timestamp TIMESTAMP, purchases_7d FLOAT, revenue_7d FLOAT)").collect()
        
        session.write_pandas(
            predictions_pd,
            "INFERENCE_RESULTS",
            database="FE_DEMO_DB",
            schema="FE_FEATURES",
            auto_create_table=True,
            overwrite=True
        )
        print(f"✅ Saved {len(predictions_for_sf)} predictions to fe_features.inference_results")
except Exception as e:
    print(f"⚠️  Could not save predictions to table: {e}")

print("\n" + "="*60)
print("📈 INFERENCE SUMMARY STATISTICS")
print("="*60)
successful_predictions = [p for p in predictions if p['status'] == 'SUCCESS']
if successful_predictions:
    probabilities = [p['churn_probability'] for p in successful_predictions]
    churn_predictions = [p['churn_prediction'] for p in successful_predictions]
    
    print(f"Total predictions: {len(successful_predictions)}")
    print(f"Customers at churn risk: {sum(churn_predictions)} ({100*sum(churn_predictions)/len(churn_predictions):.1f}%)")
    print(f"Average churn probability: {sum(probabilities)/len(probabilities):.4f}")
    print(f"Min churn probability: {min(probabilities):.4f}")
    print(f"Max churn probability: {max(probabilities):.4f}")

print("\n" + "="*60)
print("✅ INFERENCE COMPLETED!")
print("="*60)

