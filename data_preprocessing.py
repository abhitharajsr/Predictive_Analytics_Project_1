"""
Data Preprocessing Module for Telecom Customer Churn Prediction
================================================================
Handles data loading, cleaning, feature engineering, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV file."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
#  CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe:
    - Drop customerID
    - Fix TotalCharges (blank → NaN → impute)
    - Convert SeniorCitizen to categorical labels
    """
    df = df.copy()

    # Drop customer ID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # TotalCharges has blank strings — convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Impute missing TotalCharges with tenure * MonthlyCharges
    mask = df['TotalCharges'].isna()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'tenure'] * df.loc[mask, 'MonthlyCharges']

    # For tenure=0, TotalCharges would be 0 — fill any remaining NaN with 0
    df['TotalCharges'].fillna(0, inplace=True)

    # Convert SeniorCitizen from 0/1 to No/Yes for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    print(f"✅ Data cleaned. Missing values: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - AvgMonthlyCharge: TotalCharges / tenure
    - NumServices: count of subscribed services
    - TenureGroup: binned tenure into categories
    """
    df = df.copy()

    # Average monthly charge (handle divide-by-zero for tenure=0)
    df['AvgMonthlyCharge'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    # Count number of services subscribed
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    def count_services(row):
        count = 0
        for col in service_cols:
            val = str(row[col]).strip()
            if val == 'Yes' or val == 'Fiber optic' or val == 'DSL':
                count += 1
        return count

    df['NumServices'] = df.apply(count_services, axis=1)

    # Tenure grouping
    bins = [0, 12, 24, 48, 60, 72]
    labels = ['0-12', '13-24', '25-48', '49-60', '61-72']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

    print(f"✅ Feature engineering complete. New shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
#  ENCODE & SCALE
# ─────────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame, fit: bool = True, scaler=None):
    """
    Encode categorical variables and scale numerical features.
    
    Parameters:
        df: DataFrame with features + target
        fit: If True, fit a new scaler. If False, use provided scaler.
        scaler: Pre-fitted scaler (used when fit=False)
    
    Returns:
        X: feature matrix (numpy array)
        y: target vector
        feature_names: list of feature column names
        scaler: fitted StandardScaler
    """
    df = df.copy()

    # Separate target
    y = df['Churn'].map({'Yes': 1, 'No': 0}).values if 'Churn' in df.columns else None
    
    # Drop target and TenureGroup (ordinal, already captured by tenure)
    drop_cols = ['Churn']
    if 'TenureGroup' in df.columns:
        drop_cols.append('TenureGroup')
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

    # Binary encode Yes/No columns
    binary_cols = []
    for col in df.select_dtypes(include='object').columns:
        unique_vals = df[col].unique()
        if set(unique_vals).issubset({'Yes', 'No'}):
            binary_cols.append(col)
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encode remaining categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'NumServices']
    num_cols = [c for c in num_cols if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])

    # Ensure all columns are numeric
    df = df.astype(float)

    feature_names = df.columns.tolist()
    X = df.values

    print(f"✅ Encoding complete. Feature matrix shape: {X.shape}")
    return X, y, feature_names, scaler


# ─────────────────────────────────────────────
#  FULL PIPELINE
# ─────────────────────────────────────────────
def full_preprocessing_pipeline(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Run the complete preprocessing pipeline:
    Load → Clean → Engineer → Encode → Split
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler, df_clean
    """
    # Load and clean
    df = load_data(filepath)
    df = clean_data(df)
    df_clean = df.copy()  # Save clean version for EDA
    
    # Engineer features
    df = engineer_features(df)
    
    # Encode and scale
    X, y, feature_names, scaler = encode_and_scale(df, fit=True)
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set:  {X_test.shape[0]} samples")
    print(f"📊 Churn rate (train): {y_train.mean():.2%}")
    print(f"📊 Churn rate (test):  {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler, df_clean


# ─────────────────────────────────────────────
#  PREPROCESS SINGLE INPUT (for Streamlit)
# ─────────────────────────────────────────────
def preprocess_single_input(input_dict: dict, scaler, training_columns: list) -> np.ndarray:
    """
    Preprocess a single customer input for prediction.
    
    Parameters:
        input_dict: dict with raw feature values
        scaler: fitted StandardScaler
        training_columns: column names from training data
    
    Returns:
        Preprocessed feature array ready for model prediction
    """
    df = pd.DataFrame([input_dict])
    
    # Apply same transformations
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Feature engineering
    df['AvgMonthlyCharge'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )
    
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    def count_services(row):
        count = 0
        for col in service_cols:
            if col in row.index:
                val = str(row[col]).strip()
                if val in ('Yes', 'Fiber optic', 'DSL'):
                    count += 1
        return count
    
    df['NumServices'] = df.apply(count_services, axis=1)
    
    # Drop TenureGroup if present
    if 'TenureGroup' in df.columns:
        df.drop('TenureGroup', axis=1, inplace=True)
    
    # Binary encode
    for col in df.select_dtypes(include='object').columns:
        unique_vals = df[col].unique()
        if set(unique_vals).issubset({'Yes', 'No'}):
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # One-hot encode
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Align columns with training data
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]
    
    # Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'NumServices']
    num_cols = [c for c in num_cols if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])
    
    return df.values.astype(float)


if __name__ == '__main__':
    # Quick test
    filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test, features, scaler, df_clean = full_preprocessing_pipeline(filepath)
    print(f"\n✅ Pipeline test passed! Features: {len(features)}")
