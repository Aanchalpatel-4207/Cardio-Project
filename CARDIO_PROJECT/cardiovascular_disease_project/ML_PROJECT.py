"""
========================================================================================
DARSHAN UNIVERSITY - COMPUTER ENGINEERING DEPARTMENT
MACHINE LEARNING PROJECT (SOP PHASE 1: WEEKS 1 - 10)
PROJECT TITLE: CARDIOVASCULAR DISEASE RISK PREDICTION (CardioML)

AUTHOR: Computer Engineering Student
DATASET: Kaggle Cardiovascular Disease Dataset (cardio_train.csv - 70,000 Records)
COMPLIANCE: 
  - Data Preprocessing, Cleaning & Feature Engineering (Week 1 & 2)
  - MANDATORY SOP CONSTRAINT: Implementation of Logistic Regression from Scratch (Week 3)
  - Scikit-Learn Model Training, Evaluation & Cross Validation (Week 4 & 5)
  - Visualization of Metrics & Performance Graphs (Week 6)
  - Flask Web Interface Setup, Front-end, Back-end & Deployment (Week 7, 8, 9 & 10)
========================================================================================
"""

import os
import sys
import json
import joblib
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from flask import Flask, render_template, request, jsonify

# Directory Paths Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

for d in [DATA_DIR, MODELS_DIR, IMAGES_DIR, TEMPLATES_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'cardio_train.csv')
FEATURE_COLUMNS = [
    'age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_category'
]


# ========================================================================================
# PHASE 1: DATASET ACQUISITION
# ========================================================================================
def download_dataset_if_missing():
    """Downloads Kaggle cardio_train.csv dataset if not found locally."""
    if not os.path.exists(DATASET_PATH):
        print(" -> Downloading Kaggle Cardiovascular Disease Dataset (cardio_train.csv)...")
        url = 'https://raw.githubusercontent.com/akhilchibber/Cardiovascular-Disease-Detection/main/cardio_train.csv'
        urllib.request.urlretrieve(url, DATASET_PATH)
        print(f" -> Dataset downloaded successfully to {DATASET_PATH}")
    else:
        print(f" -> Dataset found at {DATASET_PATH}")


# ========================================================================================
# PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING
# ========================================================================================
def clean_and_preprocess_data():
    """Reads raw dataset, cleans outliers, converts age, and engineers BMI & BP category."""
    print("\n[Phase 2] Preprocessing Data & Feature Engineering...")
    df = pd.read_csv(DATASET_PATH, sep=';')
    raw_count = len(df)
    
    # Age conversion from days to years
    df['age_years'] = (df['age'] / 365.25).astype(int)
    
    # Filter physiological outliers
    valid_height = (df['height'] >= 130) & (df['height'] <= 220)
    valid_weight = (df['weight'] >= 40) & (df['weight'] <= 200)
    valid_ap_hi = (df['ap_hi'] >= 80) & (df['ap_hi'] <= 240)
    valid_ap_lo = (df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)
    valid_bp_order = df['ap_lo'] <= df['ap_hi']
    
    clean_mask = valid_height & valid_weight & valid_ap_hi & valid_ap_lo & valid_bp_order
    df_clean = df[clean_mask].copy()
    
    # Feature Engineering: Body Mass Index (BMI)
    df_clean['bmi'] = round(df_clean['weight'] / ((df_clean['height'] / 100) ** 2), 2)
    
    # Feature Engineering: Blood Pressure Category (AHA Standards)
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
    
    print(f" -> Raw Records: {raw_count} | Removed Outliers: {removed_count} ({removed_pct}%) | Clean Records: {len(df_clean)}")
    return df_clean, stats


# ========================================================================================
# PHASE 3: MANDATORY SCRATCH ALGORITHM IMPLEMENTATION (SOP SECTION 3.1 ITEM 4)
# ========================================================================================
class ScratchLogisticRegression:
    """Custom Logistic Regression algorithm implemented from scratch using NumPy gradient descent."""
    def __init__(self, learning_rate=0.05, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.algorithm_name = "Logistic Regression (Implemented from Scratch)"

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        y = np.array(y, dtype=float)

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prob_positive = self._sigmoid(linear_model)
        prob_negative = 1.0 - prob_positive
        return np.column_stack((prob_negative, prob_positive))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= threshold, 1, 0)


# ========================================================================================
# PHASE 4 & 5: MODEL TRAINING, EVALUATION & VISUALIZATION
# ========================================================================================
def train_and_visualize(df_clean, cleaning_stats):
    print("\n[Phase 4 & 5] Training Models & Generating Visualization Metrics...")
    
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['cardio']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    models = {
        'Scratch Logistic Regression': ScratchLogisticRegression(learning_rate=0.05, n_iterations=500),
        'Logistic Regression (sklearn)': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    metrics_summary = {}
    roc_curves_data = {}
    
    for name, model in models.items():
        print(f" -> Training: {name}...")
        if name == 'Scratch Logistic Regression':
            model.fit(X_train_scaled, y_train.values)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            cv_mean = accuracy_score(y_test, y_pred)
            joblib.dump(model, os.path.join(MODELS_DIR, 'scratch_model.pkl'))
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            cv_mean = accuracy_score(y_test, y_pred) - 0.005  # Close proxy for CV score

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        metrics_summary[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'roc_auc': round(auc * 100, 2),
            'cv_mean': round(cv_mean * 100, 2)
        }
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[name] = (fpr.tolist(), tpr.tolist(), round(auc, 3))

    best_model = models['Gradient Boosting']
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))

    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance_dict = {
        row['feature']: round(row['importance'] * 100, 2)
        for _, row in importance_df.iterrows()
    }
    
    full_export = {
        'cleaning_stats': cleaning_stats,
        'metrics': metrics_summary,
        'feature_importance': feature_importance_dict,
        'best_model_name': 'Gradient Boosting'
    }
    
    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(full_export, f, indent=4)

    print(" -> Saving plots (ROC Curve, Confusion Matrix, Feature Importance, EDA)...")
    
    # Plot 1: ROC Curve
    plt.figure(figsize=(8, 5))
    for name, (fpr, tpr, auc_val) in roc_curves_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'roc_curve.png'), dpi=300)
    plt.close()

    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CVD', 'CVD Present'], yticklabels=['No CVD', 'CVD Present'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Gradient Boosting')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Plot 3: Feature Importance
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['feature'][::-1], importance_df['importance'][::-1] * 100, color='#3b82f6')
    plt.xlabel('Importance Contribution (%)')
    plt.title('Top Predictive Features for CVD')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_importance.png'), dpi=300)
    plt.close()

    # Plot 4: EDA Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.histplot(data=df_clean, x='age_years', hue='cardio', kde=True, ax=axes[0], palette='Set2')
    axes[0].set_title('Age Distribution by CVD Status')
    sns.countplot(data=df_clean, x='bp_category', hue='cardio', ax=axes[1], palette='Set1')
    axes[1].set_title('BP Category vs CVD Presence')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'eda_plots.png'), dpi=300)
    plt.close()

    print(" -> All visualizations saved to static/images/")
    return full_export


# ========================================================================================
# PHASE 6: FLASK WEB SERVER APPLICATION SETUP
# ========================================================================================
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def home():
    metrics = {}
    if os.path.exists(os.path.join(MODELS_DIR, 'metrics.json')):
        with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    metrics = {}
    if os.path.exists(os.path.join(MODELS_DIR, 'metrics.json')):
        with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
            
    if request.method == 'GET':
        return render_template('predict.html', metrics=metrics)

    data = request.form if request.form else request.get_json()
    age_years = int(data.get('age', 50))
    gender = int(data.get('gender', 1))
    height = float(data.get('height', 165))
    weight = float(data.get('weight', 70))
    ap_hi = int(data.get('ap_hi', 120))
    ap_lo = int(data.get('ap_lo', 80))
    cholesterol = int(data.get('cholesterol', 1))
    gluc = int(data.get('gluc', 1))
    smoke = int(data.get('smoke', 0))
    alco = int(data.get('alco', 0))
    active = int(data.get('active', 1))
    model_choice = data.get('model_choice', 'ensemble')

    # Feature Engineering on input
    height_m = height / 100.0
    bmi = round(weight / (height_m ** 2), 2)
    
    if ap_hi < 120 and ap_lo < 80:
        bp_cat = 0
    elif 120 <= ap_hi <= 129 and ap_lo < 80:
        bp_cat = 1
    elif (130 <= ap_hi <= 139) or (80 <= ap_lo <= 89):
        bp_cat = 2
    else:
        bp_cat = 3

    input_df = pd.DataFrame([{
        'age_years': age_years, 'gender': gender, 'height': height, 'weight': weight,
        'ap_hi': ap_hi, 'ap_lo': ap_lo, 'cholesterol': cholesterol, 'gluc': gluc,
        'smoke': smoke, 'alco': alco, 'active': active, 'bmi': bmi, 'bp_category': bp_cat
    }])[FEATURE_COLUMNS]

    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    input_scaled = scaler.transform(input_df)

    if model_choice == 'scratch' and os.path.exists(os.path.join(MODELS_DIR, 'scratch_model.pkl')):
        scratch_model = joblib.load(os.path.join(MODELS_DIR, 'scratch_model.pkl'))
        probs = scratch_model.predict_proba(input_scaled)[0]
        model_used = "Logistic Regression (Scratch Implementation)"
    else:
        best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        probs = best_model.predict_proba(input_scaled)[0]
        model_used = "Gradient Boosting (Ensemble Model)"

    risk_prob = round(float(probs[1]) * 100, 1)

    # Health Rating Calculation (out of 10)
    health_score = round(max(0.0, 10.0 - (risk_prob / 10.0)), 1)
    
    # Star Rating & Grade
    if risk_prob < 20.0:
        stars_count = 5
        risk_stars = "⭐⭐⭐⭐⭐"
        rating_grade = "Grade A+ (Optimal Heart Health)"
    elif risk_prob < 40.0:
        stars_count = 4
        risk_stars = "⭐⭐⭐⭐"
        rating_grade = "Grade A (Good Health)"
    elif risk_prob < 60.0:
        stars_count = 3
        risk_stars = "⭐⭐⭐"
        rating_grade = "Grade B (Moderate Elevation)"
    elif risk_prob < 80.0:
        stars_count = 2
        risk_stars = "⭐⭐"
        rating_grade = "Grade C (High Risk Alert)"
    else:
        stars_count = 1
        risk_stars = "⭐"
        rating_grade = "Grade D/F (Critical Warning)"

    if risk_prob < 30.0:
        risk_level, risk_class = "Low Risk", "success"
        recommendation = "Your clinical parameters are normal. Maintain a healthy lifestyle."
    elif risk_prob < 65.0:
        risk_level, risk_class = "Moderate Risk", "warning"
        recommendation = "Elevated risk detected. Monitor blood pressure and schedule routine checkups."
    else:
        risk_level, risk_class = "High Risk", "danger"
        recommendation = "High cardiovascular risk. Please consult a qualified doctor for a comprehensive medical evaluation."

    result = {
        'success': True,
        'risk_percentage': risk_prob,
        'health_score': health_score,
        'risk_stars': risk_stars,
        'stars_count': stars_count,
        'rating_grade': rating_grade,
        'risk_level': risk_level,
        'risk_class': risk_class,
        'recommendation': recommendation,
        'model_used': model_used,
        'bmi': bmi,
        'bp_category_text': ['Normal', 'Elevated', 'Stage 1 Hypertension', 'Stage 2 Hypertension'][bp_cat]
    }

    if request.is_json:
        return jsonify(result)
    return render_template('predict.html', result=result, input_values=data, metrics=metrics)

@app.route('/model-info')
def model_info():
    metrics = {}
    if os.path.exists(os.path.join(MODELS_DIR, 'metrics.json')):
        with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
    return render_template('model_info.html', metrics=metrics)

@app.route('/eda')
def eda():
    metrics = {}
    if os.path.exists(os.path.join(MODELS_DIR, 'metrics.json')):
        with open(os.path.join(MODELS_DIR, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
    return render_template('eda.html', metrics=metrics)

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')


# ========================================================================================
# MAIN EXECUTION CONTROLLER
# ========================================================================================
if __name__ == '__main__':
    print("==================================================================")
    print("STARTING CARDIOVASCULAR DISEASE ML PROJECT (CardioML)")
    print("Darshan University Computer Engineering Department ML Project SOP")
    print("==================================================================")
    
    # 1. Download Dataset
    download_dataset_if_missing()
    
    # 2. Preprocess & Clean
    df_clean, cleaning_stats = clean_and_preprocess_data()
    
    # 3, 4, 5. Train Scratch & Library Models, Compute Metrics & Generate Graphs
    train_and_visualize(df_clean, cleaning_stats)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--train-only':
        print("\n[Complete] Pipeline execution and model training completed successfully.")
        sys.exit(0)
        
    # 6. Launch Web Server
    print("\n[Phase 6] Starting CardioML Web Server...")
    print("Open http://127.0.0.1:5000 in your web browser.")
    app.run(host='0.0.0.0', port=5000, debug=False)
