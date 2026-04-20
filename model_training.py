"""
Model Training Module for Telecom Customer Churn Prediction
=============================================================
Trains 5 ML models, handles class imbalance with SMOTE,
performs hyperparameter tuning, and saves the best model.
"""

import numpy as np
import pandas as pd
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE

# Try importing optional models
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available, skipping.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not available, skipping.")


# ─────────────────────────────────────────────
#  SMOTE OVERSAMPLING
# ─────────────────────────────────────────────
def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to balance training data."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n🔄 SMOTE Applied:")
    print(f"   Before: {np.bincount(y_train)} (No Churn / Churn)")
    print(f"   After:  {np.bincount(y_resampled)} (No Churn / Churn)")
    
    return X_resampled, y_resampled


# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models():
    """Return dict of model instances to train."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, solver='liblinear'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, random_state=42
        ),
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss',
            use_label_encoder=False
        )
    
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
    
    return models


# ─────────────────────────────────────────────
#  HYPERPARAMETER GRIDS
# ─────────────────────────────────────────────
def get_param_grids():
    """Return hyperparameter grids for tuning."""
    grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf'],
        },
    }
    
    if HAS_XGBOOST:
        grids['XGBoost'] = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
        }
    
    if HAS_LIGHTGBM:
        grids['LightGBM'] = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 50],
        }
    
    return grids


# ─────────────────────────────────────────────
#  TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_all_models(X_train, X_test, y_train, y_test, use_smote=True, tune=False):
    """
    Train all models and return results.
    
    Parameters:
        X_train, X_test: feature matrices
        y_train, y_test: target vectors
        use_smote: apply SMOTE oversampling
        tune: perform hyperparameter tuning (slower)
    
    Returns:
        results: dict of {model_name: {model, metrics, y_pred, y_prob}}
        metrics_df: DataFrame with metrics comparison
    """
    # Apply SMOTE if requested
    if use_smote:
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    models = get_models()
    param_grids = get_param_grids()
    results = {}
    all_metrics = []
    
    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  🚀 Training: {name}")
        print(f"{'─'*50}")
        
        start_time = time.time()
        
        if tune and name in param_grids:
            # Hyperparameter tuning
            print(f"  🔧 Tuning hyperparameters...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(
                model, param_grids[name],
                cv=cv, scoring='f1', n_jobs=-1, verbose=0
            )
            grid.fit(X_train_balanced, y_train_balanced)
            best_model = grid.best_estimator_
            print(f"  ✅ Best params: {grid.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train_balanced, y_train_balanced)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else 0,
            'Train Time (s)': round(train_time, 2),
        }
        
        all_metrics.append(metrics)
        results[name] = {
            'model': best_model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }
        
        # Print results
        print(f"  ⏱️  Time: {train_time:.2f}s")
        print(f"  📈 Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  📈 Precision: {metrics['Precision']:.4f}")
        print(f"  📈 Recall:    {metrics['Recall']:.4f}")
        print(f"  📈 F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"  📈 ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Determine best model
    best_idx = metrics_df['F1-Score'].idxmax()
    best_name = metrics_df.loc[best_idx, 'Model']
    print(f"\n{'='*50}")
    print(f"  🏆 BEST MODEL: {best_name}")
    print(f"     F1-Score: {metrics_df.loc[best_idx, 'F1-Score']:.4f}")
    print(f"     ROC-AUC:  {metrics_df.loc[best_idx, 'ROC-AUC']:.4f}")
    print(f"{'='*50}")
    
    return results, metrics_df


# ─────────────────────────────────────────────
#  SAVE / LOAD MODELS
# ─────────────────────────────────────────────
def save_best_model(results, metrics_df, feature_names, scaler, save_dir='models'):
    """Save the best model, scaler, and metadata."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Find best model by F1-Score
    best_idx = metrics_df['F1-Score'].idxmax()
    best_name = metrics_df.loc[best_idx, 'Model']
    best_model = results[best_name]['model']
    
    # Save model
    model_path = os.path.join(save_dir, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(save_dir, 'preprocessor.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'best_model_name': best_name,
        'feature_names': feature_names,
        'metrics': metrics_df.to_dict('records'),
    }
    metadata_path = os.path.join(save_dir, 'model_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    
    # Save all model results for comparison
    all_results_path = os.path.join(save_dir, 'all_results.pkl')
    results_to_save = {}
    for name, data in results.items():
        results_to_save[name] = {
            'model': data['model'],
            'y_pred': data['y_pred'],
            'y_prob': data['y_prob'],
        }
    joblib.dump(results_to_save, all_results_path)
    
    print(f"\n💾 Saved to '{save_dir}/':")
    print(f"   • best_model.pkl ({best_name})")
    print(f"   • preprocessor.pkl")
    print(f"   • model_metadata.pkl")
    print(f"   • all_results.pkl")
    
    return best_name, best_model


def load_model(model_dir='models'):
    """Load saved model, scaler, and metadata."""
    model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
    metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
    
    return model, scaler, metadata


if __name__ == '__main__':
    # Quick test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_preprocessing import full_preprocessing_pipeline
    
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test, features, scaler, df_clean = full_preprocessing_pipeline(filepath)
    
    results, metrics_df = train_all_models(X_train, X_test, y_train, y_test, use_smote=True, tune=False)
    
    print("\n📊 Metrics Summary:")
    print(metrics_df.to_string(index=False))
    
    best_name, best_model = save_best_model(results, metrics_df, features, scaler)
    
    # ── Compute SHAP values ──
    # Use XGBoost or Random Forest for SHAP (TreeExplainer is fast and exact)
    shap_model_name = None
    shap_model_obj = None
    for preferred in ['XGBoost', 'Random Forest', 'LightGBM']:
        if preferred in results:
            shap_model_name = preferred
            shap_model_obj = results[preferred]['model']
            break
    
    if shap_model_obj is None:
        shap_model_name = best_name
        shap_model_obj = best_model
    
    from src.shap_analysis import compute_and_save_shap
    compute_and_save_shap(shap_model_obj, X_test, features, shap_model_name)

