# ============================================================================
# PHASE 5: Model Training Using Engineered Features
# Run this in a Snowflake Python Worksheet
# ============================================================================

print("="*70)
print("PACKAGE INSTALLATION CHECK")
print("="*70)
print("\n⚠️  BEFORE RUNNING THIS CODE:")
print("   1. Look for 'Packages' button at the TOP of your notebook")
print("   2. Click 'Packages' → Add these packages:")
print("      • scikit-learn")
print("      • pandas")
print("      • numpy")
print("   3. Click 'Save' or 'Apply'")
print("   4. Wait 1-2 minutes for installation")
print("   5. Then re-run this cell")
print("="*70)
print()

packages_ok = True

try:
    import pandas as pd
    print("✅ pandas: OK")
except ImportError as e:
    print(f"❌ pandas: NOT FOUND - {e}")
    packages_ok = False

try:
    import numpy as np
    print("✅ numpy: OK")
except ImportError as e:
    print(f"❌ numpy: NOT FOUND - {e}")
    packages_ok = False

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score,
        roc_auc_score,
        roc_curve,
        auc,
        classification_report,
        confusion_matrix
    )
    print("✅ scikit-learn: OK")
except ImportError as e:
    print(f"❌ scikit-learn: NOT FOUND - {e}")
    packages_ok = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("✅ matplotlib: OK")
except ImportError as e:
    print(f"❌ matplotlib: NOT FOUND - {e}")
    packages_ok = False

if not packages_ok:
    print("\n" + "="*70)
    print("❌ REQUIRED PACKAGES NOT INSTALLED")
    print("="*70)
    print("\n📦 INSTALLATION STEPS:")
    print("   1. Find 'Packages' button at the TOP of your Snowflake notebook")
    print("      (Usually in the toolbar, may have a 📦 icon)")
    print("   2. Click 'Packages' button")
    print("   3. Click 'Add Package' or '+' button")
    print("   4. Type and add each package:")
    print("      → scikit-learn")
    print("      → pandas")
    print("      → numpy")
    print("   5. Click 'Save' or 'Apply'")
    print("   6. Wait 1-2 minutes for packages to install")
    print("   7. Re-run this cell")
    print("\n💡 TIP: If you can't find the Packages button:")
    print("   • Make sure you're in a Notebook (not SQL Worksheet)")
    print("   • Try refreshing the page")
    print("   • Check the top toolbar/menu")
    print("\n📖 For detailed guide, see: INSTALL_PACKAGES_GUIDE.md")
    print("="*70)
    raise ImportError(
        "Required packages not installed. "
        "Please install scikit-learn, pandas, and numpy using the Packages button, then re-run this cell."
    )

print("\n✅ All required packages are installed!")
print("="*70)
print()

import pickle
import tempfile
import os

try:
    _ = session
    print("✅ Using session from notebook context")
except NameError:
    try:
        import snowflake.snowpark.context as snowpark_context
        session = snowpark_context.get_active_session()
        print("✅ Got session from snowpark context")
    except Exception as e:
        print(f"⚠️  Could not get session: {e}")
        print("Please ensure you're running this in a Snowflake notebook")
        raise

session.use_warehouse("FE_WH")
session.use_database("FE_DEMO_DB")
session.use_schema("FE_FEATURES")

print("Loading training data...")
try:
    training_df = session.sql("""
        SELECT * FROM fe_features.training_set
    """).to_pandas()
    
    training_df.columns = [c.lower() for c in training_df.columns]
    
    if training_df.empty:
        raise ValueError("Training set is empty! Please run Phase 4 (04_training_set_creation.sql) first.")
    
    print(f"✅ Training set loaded successfully")
    print(f"Training set shape: {training_df.shape}")
    print(f"\nTraining set preview:")
    print(training_df.head())
    print(f"\nLabel distribution:")
    print(training_df['label'].value_counts())
    
except Exception as e:
    print(f"❌ Error loading training data: {e}")
    print("\n💡 SOLUTION:")
    print("   1. Make sure you've run Phase 4 (04_training_set_creation.sql)")
    print("   2. Verify the training_set view exists: SELECT * FROM fe_features.training_set")
    raise

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

missing_columns = [col for col in feature_columns if col not in training_df.columns]
if missing_columns:
    print(f"⚠️  Warning: Missing columns: {missing_columns}")
    print(f"Available columns: {list(training_df.columns)}")
    feature_columns = [col for col in feature_columns if col in training_df.columns]
    print(f"Using available columns: {feature_columns}")

if 'label' not in training_df.columns:
    raise ValueError("'label' column not found in training set! Please check Phase 4.")

X = training_df[feature_columns].fillna(0)
y = training_df['label'].astype(int)

if len(y.unique()) < 2:
    print(f"⚠️  Warning: Only one class found in labels: {y.unique()}")
    print("Model training will proceed, but evaluation metrics may be limited.")

print(f"\nFeature matrix shape: {X.shape}")
print(f"\nFeature statistics:")
print(X.describe())

print("\n" + "="*60)
print("SPLITTING DATA AND TRAINING MODELS")
print("="*60)

if len(X) >= 20:
    print(f"✅ Dataset size: {len(X)} samples. Using stratified 70/30 split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
elif len(X) >= 10:
    print(f"✅ Dataset size: {len(X)} samples. Using stratified 80/20 split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print(f"⚠️  Small dataset ({len(X)} samples). Using 80/20 split without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Class distribution in training set:")
print(pd.Series(y_train).value_counts())
print(f"\nClass distribution in test set:")
print(pd.Series(y_test).value_counts())

print("\n" + "-"*60)
print("Training Logistic Regression model...")
print("-"*60)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

cv = StratifiedKFold(n_splits=min(5, len(X)//2), shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='f1_weighted')

print("\n=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Cross-validation F1 (mean ± std): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

try:
    if len(set(y_test)) > 1 and len(set(y_pred_proba_lr)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba_lr)
        print(f"ROC AUC: {auc_score:.4f}")
    else:
        print("ROC AUC: N/A (requires both classes in test set)")
except Exception as e:
    print(f"ROC AUC: N/A ({e})")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

print("\n" + "-"*60)
print("Training Random Forest model...")
print("-"*60)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_weighted')

print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"Cross-validation F1 (mean ± std): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

try:
    if len(set(y_test)) > 1 and len(set(y_pred_proba_rf)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba_rf)
        print(f"ROC AUC: {auc_score:.4f}")
    else:
        print("ROC AUC: N/A (requires both classes in test set)")
except Exception as e:
    print(f"ROC AUC: N/A ({e})")

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(feature_importance)

print("\n" + "-"*60)
print("Creating model evaluation visualizations...")
print("-"*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Evaluation Metrics and Visualizations', fontsize=16, fontweight='bold')

from sklearn.metrics import ConfusionMatrixDisplay
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['No Churn', 'Churn'])
disp_lr.plot(ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('Confusion Matrix - Logistic Regression')

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No Churn', 'Churn'])
disp_rf.plot(ax=axes[0, 1], cmap='Greens')
axes[0, 1].set_title('Confusion Matrix - Random Forest')

axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
if len(set(y_test)) > 1 and len(set(y_pred_proba_lr)) > 1:
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    axes[0, 2].plot(fpr_lr, tpr_lr, 'b-', label=f'LR (AUC={roc_auc_lr:.3f})', linewidth=2)
if len(set(y_test)) > 1 and len(set(y_pred_proba_rf)) > 1:
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    axes[0, 2].plot(fpr_rf, tpr_rf, 'g-', label=f'RF (AUC={roc_auc_rf:.3f})', linewidth=2)
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curves Comparison')
axes[0, 2].legend(loc='lower right')
axes[0, 2].grid(alpha=0.3)

top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'].values)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importance (Random Forest)')
axes[1, 0].invert_yaxis()

cv_data = pd.DataFrame({
    'Logistic Regression': cv_scores_lr,
    'Random Forest': cv_scores_rf
})
axes[1, 1].boxplot(cv_data.values, labels=cv_data.columns)
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('Cross-Validation F1 Scores Distribution')
axes[1, 1].grid(alpha=0.3, axis='y')

models = ['Logistic Regression', 'Random Forest']
accuracy_scores = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)]
f1_scores = [f1_score(y_test, y_pred_lr, zero_division=0), f1_score(y_test, y_pred_rf, zero_division=0)]

x = np.arange(len(models))
width = 0.35
axes[1, 2].bar(x - width/2, accuracy_scores, width, label='Accuracy', color='skyblue')
axes[1, 2].bar(x + width/2, f1_scores, width, label='F1 Score', color='orange')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Model Performance Comparison')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(models, rotation=15, ha='right')
axes[1, 2].legend()
axes[1, 2].set_ylim([0, 1])
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()

try:
    import io
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    
    stage_name = "fe_features.model_stage"
    session.sql(f"CREATE OR REPLACE STAGE {stage_name}").collect()
    
    temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    with open(temp_img_path, 'wb') as f:
        f.write(buffer.read())
    
    session.file.put(
        local_file_name=temp_img_path,
        stage_location=f"@{stage_name}",
        overwrite=True,
        auto_compress=False
    )
    print(f"✅ Visualization saved to stage: {stage_name}/model_evaluation.png")
    
    if os.path.exists(temp_img_path):
        os.unlink(temp_img_path)
except Exception as e:
    print(f"⚠️  Could not save visualization to stage: {e}")

print("✅ Visualizations created successfully")

print("\n" + "-"*60)
print("SAVING MODELS AND METRICS TO SNOWFLAKE STAGE")
print("-"*60)

try:
    stage_name = "fe_features.model_stage"
    session.sql(f"CREATE OR REPLACE STAGE {stage_name}").collect()
    print(f"✅ Stage created: {stage_name}")
    
    lr_model_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(lr_model, f)
            lr_model_path = f.name
        
        session.file.put(
            local_file_name=lr_model_path,
            stage_location=f"@{stage_name}/lr_model.pkl",
            overwrite=True
        )
        print(f"✅ Logistic Regression model saved")
        
    except Exception as e:
        print(f"⚠️  Error saving LR model: {e}")
    finally:
        if lr_model_path and os.path.exists(lr_model_path):
            try:
                os.unlink(lr_model_path)
            except Exception as e:
                pass
    
    rf_model_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(rf_model, f)
            rf_model_path = f.name
        
        session.file.put(
            local_file_name=rf_model_path,
            stage_location=f"@{stage_name}/rf_model.pkl",
            overwrite=True
        )
        print(f"✅ Random Forest model saved")
        
    except Exception as e:
        print(f"⚠️  Error saving RF model: {e}")
    finally:
        if rf_model_path and os.path.exists(rf_model_path):
            try:
                os.unlink(rf_model_path)
            except Exception as e:
                pass
    
    metrics_summary = {
        'training_date': str(pd.Timestamp.now()),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': len(feature_columns),
        'feature_names': feature_columns,
        'logistic_regression': {
            'accuracy': float(accuracy_score(y_test, y_pred_lr)),
            'precision': float(precision_score(y_test, y_pred_lr, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_lr, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_lr, zero_division=0)),
            'cv_f1_mean': float(cv_scores_lr.mean()),
            'cv_f1_std': float(cv_scores_lr.std())
        },
        'random_forest': {
            'accuracy': float(accuracy_score(y_test, y_pred_rf)),
            'precision': float(precision_score(y_test, y_pred_rf, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_rf, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_rf, zero_division=0)),
            'cv_f1_mean': float(cv_scores_rf.mean()),
            'cv_f1_std': float(cv_scores_rf.std())
        }
    }
    
    import json
    metrics_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w').name
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    session.file.put(
        local_file_name=metrics_path,
        stage_location=f"@{stage_name}/model_metrics.json",
        overwrite=True
    )
    print(f"✅ Metrics summary saved")
    
    if os.path.exists(metrics_path):
        os.unlink(metrics_path)
                
except Exception as e:
    print(f"⚠️  Error creating stage: {e}")
    print("Model training completed, but models were not saved to stage.")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED!")
print("="*60)

print("\n📊 TRAINING SUMMARY:")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Test samples: {len(X_test)}")
print(f"   • Features used: {len(feature_columns)}")
print(f"   • Models trained: Logistic Regression, Random Forest")

print("\n📈 LOGISTIC REGRESSION PERFORMANCE:")
print(f"   • Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"   • Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"   • Recall: {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"   • F1 Score: {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"   • CV F1 Score (mean ± std): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

print("\n📈 RANDOM FOREST PERFORMANCE:")
print(f"   • Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"   • Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"   • Recall: {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"   • F1 Score: {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"   • CV F1 Score (mean ± std): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

print("\n🎯 TOP 5 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

print("\n💡 NEXT STEPS:")
print("   1. Run 06_online_inference.py for predictions on new data")
print("   2. Run 07_feature_refresh.sql to refresh features regularly")
print("   3. Models available as 'lr_model' and 'rf_model' for this session")
    print("\n" + "="*70)
    print("❌ REQUIRED PACKAGES NOT INSTALLED")
    print("="*70)
    print("\n📦 INSTALLATION STEPS:")
    print("   1. Find 'Packages' button at the TOP of your Snowflake notebook")
    print("      (Usually in the toolbar, may have a 📦 icon)")
    print("   2. Click 'Packages' button")
    print("   3. Click 'Add Package' or '+' button")
    print("   4. Type and add each package:")
    print("      → scikit-learn")
    print("      → pandas")
    print("      → numpy")
    print("   5. Click 'Save' or 'Apply'")
    print("   6. Wait 1-2 minutes for packages to install")
    print("   7. Re-run this cell")
    print("\n💡 TIP: If you can't find the Packages button:")
    print("   • Make sure you're in a Notebook (not SQL Worksheet)")
    print("   • Try refreshing the page")
    print("   • Check the top toolbar/menu")
    print("\n📖 For detailed guide, see: INSTALL_PACKAGES_GUIDE.md")
    print("="*70)
    raise ImportError(
        "Required packages not installed. "
        "Please install scikit-learn, pandas, and numpy using the Packages button, then re-run this cell."
    )

print("\n✅ All required packages are installed!")
print("="*70)
print()

import pickle
import tempfile
import os

# Get session - In Snowflake notebooks, session should be available
try:
    # Try to use session directly (it should be available in notebooks)
    _ = session
    print("✅ Using session from notebook context")
except NameError:
    # If session is not found, try to get it from context
    try:
        import snowflake.snowpark.context as snowpark_context
        session = snowpark_context.get_active_session()
        print("✅ Got session from snowpark context")
    except Exception as e:
        print(f"⚠️  Could not get session: {e}")
        print("Please ensure you're running this in a Snowflake notebook")
        raise

# Set context
session.use_warehouse("FE_WH")
session.use_database("FE_DEMO_DB")
session.use_schema("FE_FEATURES")

# Step 15: Load Training Data
print("Loading training data...")
try:
    training_df = session.sql("""
        SELECT * FROM fe_features.training_set
    """).to_pandas()
    
    # Normalize column names to lowercase for consistency
    training_df.columns = [c.lower() for c in training_df.columns]
    
    if training_df.empty:
        raise ValueError("Training set is empty! Please run Phase 4 (04_training_set_creation.sql) first.")
    
    print(f"✅ Training set loaded successfully")
    print(f"Training set shape: {training_df.shape}")
    print(f"\nTraining set preview:")
    print(training_df.head())
    print(f"\nLabel distribution:")
    print(training_df['label'].value_counts())
    
except Exception as e:
    print(f"❌ Error loading training data: {e}")
    print("\n💡 SOLUTION:")
    print("   1. Make sure you've run Phase 4 (04_training_set_creation.sql)")
    print("   2. Verify the training_set view exists: SELECT * FROM fe_features.training_set")
    raise

# Step 16: Prepare Features and Labels
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

# Validate that all feature columns exist
missing_columns = [col for col in feature_columns if col not in training_df.columns]
if missing_columns:
    print(f"⚠️  Warning: Missing columns: {missing_columns}")
    print(f"Available columns: {list(training_df.columns)}")
    # Use only available columns
    feature_columns = [col for col in feature_columns if col in training_df.columns]
    print(f"Using available columns: {feature_columns}")

# Validate label column exists
if 'label' not in training_df.columns:
    raise ValueError("'label' column not found in training set! Please check Phase 4.")

X = training_df[feature_columns].fillna(0)
y = training_df['label'].astype(int)

# Check if we have both classes
if len(y.unique()) < 2:
    print(f"⚠️  Warning: Only one class found in labels: {y.unique()}")
    print("Model training will proceed, but evaluation metrics may be limited.")

print(f"\nFeature matrix shape: {X.shape}")
print(f"\nFeature statistics:")
print(X.describe())

# Step 17: Train-Test Split and Model Training
print("\n" + "="*60)
print("SPLITTING DATA AND TRAINING MODELS")
print("="*60)

# Use stratified split with adequate data
if len(X) >= 20:
    print(f"✅ Dataset size: {len(X)} samples. Using stratified 70/30 split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
elif len(X) >= 10:
    print(f"✅ Dataset size: {len(X)} samples. Using stratified 80/20 split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print(f"⚠️  Small dataset ({len(X)} samples). Using 80/20 split without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Class distribution in training set:")
print(pd.Series(y_train).value_counts())
print(f"\nClass distribution in test set:")
print(pd.Series(y_test).value_counts())

# Train Logistic Regression model
print("\n" + "-"*60)
print("Training Logistic Regression model...")
print("-"*60)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Cross-validation
cv = StratifiedKFold(n_splits=min(5, len(X)//2), shuffle=True, random_state=42)
cv_scores_lr = cross_val_score(lr_model, X_train, y_train, cv=cv, scoring='f1_weighted')

print("\n=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"Cross-validation F1 (mean ± std): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

try:
    if len(set(y_test)) > 1 and len(set(y_pred_proba_lr)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba_lr)
        print(f"ROC AUC: {auc_score:.4f}")
    else:
        print("ROC AUC: N/A (requires both classes in test set)")
except Exception as e:
    print(f"ROC AUC: N/A ({e})")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Step 18: Train Random Forest Model
print("\n" + "-"*60)
print("Training Random Forest model...")
print("-"*60)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Cross-validation
cv_scores_rf = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_weighted')

print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"Cross-validation F1 (mean ± std): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

try:
    if len(set(y_test)) > 1 and len(set(y_pred_proba_rf)) > 1:
        auc_score = roc_auc_score(y_test, y_pred_proba_rf)
        print(f"ROC AUC: {auc_score:.4f}")
    else:
        print("ROC AUC: N/A (requires both classes in test set)")
except Exception as e:
    print(f"ROC AUC: N/A ({e})")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(feature_importance)

# Step 18.5: Create Visualizations
print("\n" + "-"*60)
print("Creating model evaluation visualizations...")
print("-"*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Evaluation Metrics and Visualizations', fontsize=16, fontweight='bold')

# 1. Confusion Matrix - Logistic Regression
from sklearn.metrics import ConfusionMatrixDisplay
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['No Churn', 'Churn'])
disp_lr.plot(ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('Confusion Matrix - Logistic Regression')

# 2. Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No Churn', 'Churn'])
disp_rf.plot(ax=axes[0, 1], cmap='Greens')
axes[0, 1].set_title('Confusion Matrix - Random Forest')

# 3. ROC Curves
axes[0, 2].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
if len(set(y_test)) > 1 and len(set(y_pred_proba_lr)) > 1:
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    axes[0, 2].plot(fpr_lr, tpr_lr, 'b-', label=f'LR (AUC={roc_auc_lr:.3f})', linewidth=2)
if len(set(y_test)) > 1 and len(set(y_pred_proba_rf)) > 1:
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    axes[0, 2].plot(fpr_rf, tpr_rf, 'g-', label=f'RF (AUC={roc_auc_rf:.3f})', linewidth=2)
axes[0, 2].set_xlabel('False Positive Rate')
axes[0, 2].set_ylabel('True Positive Rate')
axes[0, 2].set_title('ROC Curves Comparison')
axes[0, 2].legend(loc='lower right')
axes[0, 2].grid(alpha=0.3)

# 4. Feature Importance
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'].values)
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importance (Random Forest)')
axes[1, 0].invert_yaxis()

# 5. Cross-validation scores comparison
cv_data = pd.DataFrame({
    'Logistic Regression': cv_scores_lr,
    'Random Forest': cv_scores_rf
})
axes[1, 1].boxplot(cv_data.values, labels=cv_data.columns)
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('Cross-Validation F1 Scores Distribution')
axes[1, 1].grid(alpha=0.3, axis='y')

# 6. Model Performance Comparison
models = ['Logistic Regression', 'Random Forest']
accuracy_scores = [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_rf)]
f1_scores = [f1_score(y_test, y_pred_lr, zero_division=0), f1_score(y_test, y_pred_rf, zero_division=0)]

x = np.arange(len(models))
width = 0.35
axes[1, 2].bar(x - width/2, accuracy_scores, width, label='Accuracy', color='skyblue')
axes[1, 2].bar(x + width/2, f1_scores, width, label='F1 Score', color='orange')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Model Performance Comparison')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(models, rotation=15, ha='right')
axes[1, 2].legend()
axes[1, 2].set_ylim([0, 1])
axes[1, 2].grid(alpha=0.3, axis='y')

plt.tight_layout()

# Save visualization to Snowflake stage
try:
    import io
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    
    # Create stage and upload
    stage_name = "fe_features.model_stage"
    session.sql(f"CREATE OR REPLACE STAGE {stage_name}").collect()
    
    temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    with open(temp_img_path, 'wb') as f:
        f.write(buffer.read())
    
    session.file.put(
        local_file_name=temp_img_path,
        stage_location=f"@{stage_name}",
        overwrite=True,
        auto_compress=False
    )
    print(f"✅ Visualization saved to stage: {stage_name}/model_evaluation.png")
    
    if os.path.exists(temp_img_path):
        os.unlink(temp_img_path)
except Exception as e:
    print(f"⚠️  Could not save visualization to stage: {e}")

print("✅ Visualizations created successfully")

# Step 19: Save Model to Snowflake Stage
print("\n" + "-"*60)
print("SAVING MODELS AND METRICS TO SNOWFLAKE STAGE")
print("-"*60)

try:
    stage_name = "fe_features.model_stage"
    session.sql(f"CREATE OR REPLACE STAGE {stage_name}").collect()
    print(f"✅ Stage created: {stage_name}")
    
    # Save Logistic Regression model
    lr_model_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(lr_model, f)
            lr_model_path = f.name
        
        session.file.put(
            local_file_name=lr_model_path,
            stage_location=f"@{stage_name}/lr_model.pkl",
            overwrite=True
        )
        print(f"✅ Logistic Regression model saved")
        
    except Exception as e:
        print(f"⚠️  Error saving LR model: {e}")
    finally:
        if lr_model_path and os.path.exists(lr_model_path):
            try:
                os.unlink(lr_model_path)
            except Exception as e:
                pass
    
    # Save Random Forest model
    rf_model_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(rf_model, f)
            rf_model_path = f.name
        
        session.file.put(
            local_file_name=rf_model_path,
            stage_location=f"@{stage_name}/rf_model.pkl",
            overwrite=True
        )
        print(f"✅ Random Forest model saved")
        
    except Exception as e:
        print(f"⚠️  Error saving RF model: {e}")
    finally:
        if rf_model_path and os.path.exists(rf_model_path):
            try:
                os.unlink(rf_model_path)
            except Exception as e:
                pass
    
    # Save metrics summary as JSON
    metrics_summary = {
        'training_date': str(pd.Timestamp.now()),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features_used': len(feature_columns),
        'feature_names': feature_columns,
        'logistic_regression': {
            'accuracy': float(accuracy_score(y_test, y_pred_lr)),
            'precision': float(precision_score(y_test, y_pred_lr, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_lr, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_lr, zero_division=0)),
            'cv_f1_mean': float(cv_scores_lr.mean()),
            'cv_f1_std': float(cv_scores_lr.std())
        },
        'random_forest': {
            'accuracy': float(accuracy_score(y_test, y_pred_rf)),
            'precision': float(precision_score(y_test, y_pred_rf, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_rf, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred_rf, zero_division=0)),
            'cv_f1_mean': float(cv_scores_rf.mean()),
            'cv_f1_std': float(cv_scores_rf.std())
        }
    }
    
    import json
    metrics_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w').name
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    session.file.put(
        local_file_name=metrics_path,
        stage_location=f"@{stage_name}/model_metrics.json",
        overwrite=True
    )
    print(f"✅ Metrics summary saved")
    
    if os.path.exists(metrics_path):
        os.unlink(metrics_path)
                
except Exception as e:
    print(f"⚠️  Error creating stage: {e}")
    print("Model training completed, but models were not saved to stage.")

print("\n" + "="*60)
print("MODEL TRAINING COMPLETED!")
print("="*60)

# Comprehensive Summary
print("\n📊 TRAINING SUMMARY:")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Test samples: {len(X_test)}")
print(f"   • Features used: {len(feature_columns)}")
print(f"   • Models trained: Logistic Regression, Random Forest")

print("\n📈 LOGISTIC REGRESSION PERFORMANCE:")
print(f"   • Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"   • Precision: {precision_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"   • Recall: {recall_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"   • F1 Score: {f1_score(y_test, y_pred_lr, zero_division=0):.4f}")
print(f"   • CV F1 Score (mean ± std): {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

print("\n📈 RANDOM FOREST PERFORMANCE:")
print(f"   • Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"   • Precision: {precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"   • Recall: {recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"   • F1 Score: {f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
print(f"   • CV F1 Score (mean ± std): {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

print("\n🎯 TOP 5 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

print("\n💡 NEXT STEPS:")
print("   1. Run 06_online_inference.py for predictions on new data")
print("   2. Run 07_feature_refresh.sql to refresh features regularly")
print("   3. Models available as 'lr_model' and 'rf_model' for this session")

