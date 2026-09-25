"""
Cardiovascular Disease Prediction - Data Preprocessing & Feature Engineering
Following Darshan University ML Project SOP - Week 2 requirements.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_category'
]

def load_raw_data(file_path):
    """Loads dataset from CSV file using semicolon separator."""
    df = pd.read_csv(file_path, sep=';')
    return df

def clean_data(df):
    """
    Cleans raw dataset by handling missing values, filtering physiological outliers,
    and performing feature engineering (Age in years, BMI, Blood Pressure Category).
    """
    raw_count = len(df)
    
    # 1. Age conversion from days to years
    df = df.copy()
    df['age_years'] = (df['age'] / 365.25).astype(int)
    
    # 2. Filter physiological outliers
    valid_height = (df['height'] >= 130) & (df['height'] <= 220)
    valid_weight = (df['weight'] >= 40) & (df['weight'] <= 200)
    valid_ap_hi = (df['ap_hi'] >= 80) & (df['ap_hi'] <= 240)
    valid_ap_lo = (df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)
    valid_bp_order = df['ap_lo'] <= df['ap_hi']
    
    clean_mask = valid_height & valid_weight & valid_ap_hi & valid_ap_lo & valid_bp_order
    df_clean = df[clean_mask].copy()
    
    # 3. Feature Engineering
    # BMI = weight (kg) / (height (m))^2
    df_clean['bmi'] = round(df_clean['weight'] / ((df_clean['height'] / 100) ** 2), 2)
    
    # Blood Pressure Category (AHA guidelines)
    def categorize_bp(row):
        hi, lo = row['ap_hi'], row['ap_lo']
        if hi < 120 and lo < 80:
            return 0  # Normal
        elif 120 <= hi <= 129 and lo < 80:
            return 1  # Elevated
        elif (130 <= hi <= 139) or (80 <= lo <= 89):
            return 2  # Stage 1 Hypertension
        else:
            return 3  # Stage 2 Hypertension
            
    df_clean['bp_category'] = df_clean.apply(categorize_bp, axis=1)
    
    removed_count = raw_count - len(df_clean)
    removed_pct = round((removed_count / raw_count) * 100, 2)
    
    stats = {
        'raw_records': raw_count,
        'removed_records': removed_count,
        'removed_percentage': removed_pct,
        'final_records': len(df_clean)
    }
    
    return df_clean, stats

def get_features_and_target(df):
    """Extracts feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS]
    y = df['cardio']
    return X, y

def scale_features(X_train, X_test):
    """Fits StandardScaler on X_train and transforms X_train and X_test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == '__main__':
    data_path = r'C:\Users\HP\.gemini\antigravity\scratch\cardiovascular_disease_project\data\cardio_train.csv'
    df = load_raw_data(data_path)
    df_clean, stats = clean_data(df)
    print("Pre-processing Complete!")
    print("Stats:", stats)
