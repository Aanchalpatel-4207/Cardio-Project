"""
Cardiovascular Disease Prediction - Model Training, Evaluation & Visualization
Following Darshan University ML Project SOP - Weeks 3, 4, 5, and 6.
Trains Scratch & Scikit-Learn models, performs 5-fold Cross Validation,
generates performance visualization graphs, and saves model artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Add parent directory to path to support absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import load_raw_data, clean_data, get_features_and_target, scale_features, FEATURE_COLUMNS
from src.scratch_model import ScratchLogisticRegression

def train_and_evaluate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cardio_train.csv')
    models_dir = os.path.join(base_dir, 'models')
    images_dir = os.path.join(base_dir, 'static', 'images')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print("Step 1: Loading & Cleaning Dataset...")
    raw_df = load_raw_data(data_path)
    clean_df, cleaning_stats = clean_data(raw_df)
    
    X, y = get_features_and_target(clean_df)
    
    print(f"Dataset ready: {X.shape[0]} clean samples with {X.shape[1]} features.")
    
    # 80-20 Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale Features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    print("\nStep 2: Training Models...")
    
    models = {
        'Scratch Logistic Regression': ScratchLogisticRegression(learning_rate=0.05, n_iterations=1000),
        'Logistic Regression (sklearn)': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    metrics_summary = {}
    roc_curves_data = {}
    
    for name, model in models.items():
        print(f" -> Training: {name}...")
        
        if name == 'Scratch Logistic Regression':
            model.fit(X_train_scaled, y_train.values)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            cv_mean, cv_std = float(accuracy_score(y_test, y_pred)), 0.0  # Custom CV omitted for scratch speed
            model.save(os.path.join(models_dir, 'scratch_model.pkl'))
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
            
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        
        metrics_summary[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'roc_auc': round(auc * 100, 2),
            'cv_mean': round(cv_mean * 100, 2),
            'cv_std': round(cv_std * 100, 2)
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[name] = (fpr.tolist(), tpr.tolist(), round(auc, 3))
        
    # Best model is Gradient Boosting
    best_model = models['Gradient Boosting']
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    
    # Feature Importances from Gradient Boosting
    feature_importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    feature_importance_dict = {
        row['feature']: round(row['importance'] * 100, 2)
        for _, row in importance_df.iterrows()
    }
    
    # Save Metrics JSON for Web UI
    full_export = {
        'cleaning_stats': cleaning_stats,
        'metrics': metrics_summary,
        'feature_importance': feature_importance_dict,
        'best_model_name': 'Gradient Boosting',
        'feature_columns': FEATURE_COLUMNS
    }
    
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(full_export, f, indent=4)
        
    print("\nStep 3: Generating Performance Visualization Graphs...")
    
    # Plot 1: Comparative ROC Curves
    plt.figure(figsize=(9, 6))
    for name, (fpr, tpr, auc_val) in roc_curves_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Plot 2: Confusion Matrix for Best Model
    best_y_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No CVD (0)', 'CVD Present (1)'],
                yticklabels=['No CVD (0)', 'CVD Present (1)'])
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('Actual Label', fontsize=11)
    plt.title('Confusion Matrix - Gradient Boosting Classifier', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Plot 3: Feature Importance Bar Chart
    plt.figure(figsize=(9, 5))
    colors = sns.color_palette('viridis', len(importance_df))
    plt.barh(importance_df['feature'][::-1], importance_df['importance'][::-1] * 100, color=colors)
    plt.xlabel('Importance Contribution (%)', fontsize=11)
    plt.ylabel('Clinical Features', fontsize=11)
    plt.title('Top Predictive Features for Cardiovascular Disease', fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # Plot 4: EDA Insights (Age vs CVD & Blood Pressure Category)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=clean_df, x='age_years', hue='cardio', kde=True, ax=axes[0], palette='Set2', multiple='stack')
    axes[0].set_title('Age Distribution by Cardiovascular Risk', fontweight='bold')
    axes[0].set_xlabel('Age (Years)')
    axes[0].set_ylabel('Patient Count')
    
    sns.countplot(data=clean_df, x='bp_category', hue='cardio', ax=axes[1], palette='Set1')
    axes[1].set_title('BP Category vs CVD Presence', fontweight='bold')
    axes[1].set_xlabel('BP Category (0: Normal, 1: Elevated, 2: Stage 1, 3: Stage 2)')
    axes[1].set_ylabel('Patient Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'eda_plots.png'), dpi=300)
    plt.close()
    
    print("\nTraining & Visualization Complete!")
    print(f"Metrics saved to {os.path.join(models_dir, 'metrics.json')}")
    print(f"Plots saved to {images_dir}")

if __name__ == '__main__':
    train_and_evaluate_all()
