"""
SHAP Analysis Module for Telecom Customer Churn Prediction
============================================================
Provides SHAP (SHapley Additive exPlanations) for model interpretability.
Answers: WHY does the model predict a customer will churn?
"""

import numpy as np
import pandas as pd
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  COMPUTE SHAP VALUES
# ─────────────────────────────────────────────
def compute_shap_values(model, X_data, feature_names, model_name="Model"):
    """
    Compute SHAP values for a given model.
    
    Parameters:
        model: trained model object
        X_data: feature matrix (numpy array or DataFrame)
        feature_names: list of feature column names
        model_name: name of the model (for logging)
    
    Returns:
        shap_values: SHAP values object
        explainer: SHAP explainer object
    """
    print(f"\n🧠 Computing SHAP values for {model_name}...")
    
    # Convert to DataFrame for readable feature names
    if isinstance(X_data, np.ndarray):
        X_df = pd.DataFrame(X_data, columns=feature_names)
    else:
        X_df = X_data.copy()
    
    # Choose explainer based on model type
    model_type_name = type(model).__name__
    
    if model_type_name in ('XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier',
                           'GradientBoostingClassifier'):
        # TreeExplainer is fast and exact for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_df)
    else:
        # KernelExplainer for model-agnostic SHAP (slower)
        # Use a background sample to speed things up
        background = shap.sample(X_df, min(100, len(X_df)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer(X_df[:200])  # Limit for speed
    
    print(f"   ✅ SHAP computation complete for {model_name}")
    return shap_values, explainer


def compute_and_save_shap(model, X_test, feature_names, model_name, save_dir='models'):
    """Compute SHAP values and save them to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    shap_values, explainer = compute_shap_values(model, X_test, feature_names, model_name)
    
    # Save SHAP values
    shap_path = os.path.join(save_dir, 'shap_values.pkl')
    joblib.dump({
        'shap_values': shap_values,
        'feature_names': feature_names,
        'model_name': model_name,
        'X_test': X_test,
    }, shap_path)
    
    print(f"   💾 SHAP values saved to {shap_path}")
    return shap_values, explainer


def load_shap_values(model_dir='models'):
    """Load saved SHAP values from disk."""
    shap_path = os.path.join(model_dir, 'shap_values.pkl')
    if os.path.exists(shap_path):
        data = joblib.load(shap_path)
        return data['shap_values'], data['feature_names'], data['model_name'], data['X_test']
    return None, None, None, None


# ─────────────────────────────────────────────
#  SHAP-BASED INSIGHTS
# ─────────────────────────────────────────────
def get_top_churn_drivers(shap_values, feature_names, top_n=10):
    """
    Extract the top N features driving churn predictions.
    
    Returns:
        DataFrame with Feature, Mean |SHAP|, Direction
    """
    # Handle different SHAP output formats
    if hasattr(shap_values, 'values'):
        vals = shap_values.values
    else:
        vals = shap_values
    
    # If multi-output (binary classification), take the positive class
    if len(vals.shape) == 3:
        vals = vals[:, :, 1]
    
    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(vals).mean(axis=0)
    mean_shap = vals.mean(axis=0)
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap,
        'Mean SHAP': mean_shap,
        'Direction': ['Increases Churn ↑' if v > 0 else 'Decreases Churn ↓' for v in mean_shap],
    })
    
    df = df.sort_values('Mean |SHAP|', ascending=False).head(top_n).reset_index(drop=True)
    return df


def generate_retention_insights(shap_values, feature_names, X_test):
    """
    Generate actionable customer retention insights from SHAP analysis.
    
    Returns:
        list of insight dicts: {title, description, action, impact}
    """
    drivers = get_top_churn_drivers(shap_values, feature_names, top_n=20)
    
    insights = []
    
    # Map feature names to business insights
    feature_insight_map = {
        'tenure': {
            'title': '⏱️ Customer Tenure — #1 Retention Factor',
            'description': 'Longer tenure strongly reduces churn. New customers (0-12 months) are the most at-risk segment.',
            'action': 'Launch an onboarding program with milestone rewards at 3, 6, and 12 months to build loyalty early.',
            'impact': 'HIGH',
        },
        'Contract_Two year': {
            'title': '📋 Two-Year Contracts Lock In Loyalty',
            'description': 'Customers on two-year contracts have dramatically lower churn (~3%) compared to month-to-month (~42%).',
            'action': 'Offer incentives (discounts, free upgrades) for month-to-month customers to switch to annual or two-year contracts.',
            'impact': 'HIGH',
        },
        'Contract_One year': {
            'title': '📋 One-Year Contracts Reduce Risk',
            'description': 'Annual contracts are a middle ground — lower churn than month-to-month but not as strong as two-year.',
            'action': 'Position one-year contracts as a stepping stone with easy upgrade paths to two-year plans.',
            'impact': 'MEDIUM',
        },
        'MonthlyCharges': {
            'title': '💰 High Monthly Charges Drive Churn',
            'description': 'Customers paying higher monthly bills are more likely to churn — indicating price sensitivity.',
            'action': 'Create value-based bundles that reduce perceived cost. Offer loyalty discounts for high-paying long-term customers.',
            'impact': 'HIGH',
        },
        'TotalCharges': {
            'title': '💳 Total Spend Indicates Commitment',
            'description': 'Higher total charges (from longer tenure) correlate with lower churn — these customers have "invested" in the service.',
            'action': 'Highlight accumulated savings and tenure milestones in customer communications.',
            'impact': 'MEDIUM',
        },
        'InternetService_Fiber optic': {
            'title': '🌐 Fiber Optic Customers Churn More',
            'description': 'Fiber optic users show ~42% churn rate vs ~19% for DSL — possibly due to premium pricing or unmet expectations.',
            'action': 'Audit fiber optic service quality. Offer satisfaction guarantees or speed upgrades to justify the premium.',
            'impact': 'HIGH',
        },
        'OnlineSecurity': {
            'title': '🔒 Online Security Reduces Churn',
            'description': 'Customers without online security are significantly more likely to churn.',
            'action': 'Bundle online security as a free add-on for at-risk customers, especially new fiber optic subscribers.',
            'impact': 'MEDIUM',
        },
        'TechSupport': {
            'title': '🛠️ Tech Support Is a Retention Tool',
            'description': 'Customers without tech support have higher churn — support builds trust and resolves friction.',
            'action': 'Offer free tech support for the first 6 months. Flag support-less customers for proactive outreach.',
            'impact': 'MEDIUM',
        },
        'PaymentMethod_Electronic check': {
            'title': '💳 Electronic Check = Flight Risk',
            'description': 'Electronic check users churn at ~45% — much higher than auto-pay methods (~15-18%).',
            'action': 'Incentivize migration to auto-pay (bank transfer/credit card) with a one-time discount or bill credit.',
            'impact': 'HIGH',
        },
        'PaperlessBilling': {
            'title': '📱 Paperless Billing Correlates with Churn',
            'description': 'Paperless billing customers churn more — possibly because they are less engaged or more tech-savvy (more options).',
            'action': 'Increase engagement for paperless customers with personalized emails, app notifications, and loyalty offers.',
            'impact': 'LOW',
        },
        'NumServices': {
            'title': '📦 Service Bundling Reduces Churn',
            'description': 'Customers using more services have lower churn — each additional service increases switching cost.',
            'action': 'Cross-sell and upsell add-on services (backup, security, streaming) to single-service customers.',
            'impact': 'HIGH',
        },
        'SeniorCitizen': {
            'title': '👴 Senior Citizens Need Special Attention',
            'description': 'Senior citizens have higher churn rates (~41%) compared to non-seniors (~24%).',
            'action': 'Create senior-specific plans with simplified billing, dedicated support lines, and personalized pricing.',
            'impact': 'MEDIUM',
        },
        'Dependents': {
            'title': '👨‍👩‍👧‍👦 Family Customers Are Stickier',
            'description': 'Customers with dependents are less likely to churn — family ties make switching harder.',
            'action': 'Market family plans and multi-line discounts to single-account customers.',
            'impact': 'LOW',
        },
        'Partner': {
            'title': '💑 Partnered Customers Stay Longer',
            'description': 'Customers with partners have lower churn — shared accounts add inertia.',
            'action': 'Offer partner/household discounts and shared data plans.',
            'impact': 'LOW',
        },
    }
    
    # Match top SHAP drivers to business insights
    for _, row in drivers.iterrows():
        feature = row['Feature']
        if feature in feature_insight_map:
            insight = feature_insight_map[feature].copy()
            insight['shap_importance'] = row['Mean |SHAP|']
            insight['direction'] = row['Direction']
            insights.append(insight)
    
    # Sort by SHAP importance
    insights.sort(key=lambda x: x['shap_importance'], reverse=True)
    
    return insights


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_preprocessing import full_preprocessing_pipeline
    from src.model_training import load_model
    
    # Load data
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test, features, scaler, df_clean = full_preprocessing_pipeline(filepath)
    
    # Load model
    model, _, metadata = load_model()
    model_name = metadata['best_model_name']
    
    # Compute SHAP
    shap_values, explainer = compute_and_save_shap(model, X_test, features, model_name)
    
    # Print top drivers
    drivers = get_top_churn_drivers(shap_values, features)
    print("\n🏆 Top 10 Churn Drivers (by SHAP):")
    print(drivers.to_string(index=False))
    
    # Generate insights
    insights = generate_retention_insights(shap_values, features, X_test)
    print(f"\n💡 Generated {len(insights)} actionable retention insights")
    for i, ins in enumerate(insights[:5], 1):
        print(f"\n  {i}. {ins['title']}")
        print(f"     Impact: {ins['impact']}")
        print(f"     Action: {ins['action']}")
