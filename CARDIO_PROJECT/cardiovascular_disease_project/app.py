"""
Cardiovascular Disease Risk Assessment - Flask Backend Application (CardioML)
Fulfills Darshan University ML Project SOP - Weeks 7, 8, and 9.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.preprocessing import FEATURE_COLUMNS
from src.scratch_model import ScratchLogisticRegression

app = Flask(__name__)

# Paths
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
SCRATCH_MODEL_PATH = os.path.join(MODELS_DIR, 'scratch_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')

# Global variables for loaded models
best_model = None
scratch_model = None
scaler = None
metrics_data = {}

def load_resources():
    global best_model, scratch_model, scaler, metrics_data
    if os.path.exists(BEST_MODEL_PATH):
        best_model = joblib.load(BEST_MODEL_PATH)
    if os.path.exists(SCRATCH_MODEL_PATH):
        scratch_model = ScratchLogisticRegression.load(SCRATCH_MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)

# Load artifacts on launch
load_resources()

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 2)

def categorize_bp(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return 0  # Normal
    elif 120 <= ap_hi <= 129 and ap_lo < 80:
        return 1  # Elevated
    elif (130 <= ap_hi <= 139) or (80 <= ap_lo <= 89):
        return 2  # Stage 1 Hypertension
    else:
        return 3  # Stage 2 Hypertension

@app.route('/')
def home():
    """Landing Page / Hero Section."""
    return render_template('index.html', metrics=metrics_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction Form & Assessment Endpoint."""
    if request.method == 'GET':
        return render_template('predict.html', metrics=metrics_data)

    try:
        # Extract inputs from form or JSON
        data = request.form if request.form else request.get_json()
        
        age_years = int(data.get('age', 50))
        gender = int(data.get('gender', 1))  # 1: female, 2: male
        height = float(data.get('height', 165))
        weight = float(data.get('weight', 70))
        ap_hi = int(data.get('ap_hi', 120))
        ap_lo = int(data.get('ap_lo', 80))
        cholesterol = int(data.get('cholesterol', 1))  # 1, 2, 3
        gluc = int(data.get('gluc', 1))                # 1, 2, 3
        smoke = int(data.get('smoke', 0))              # 0, 1
        alco = int(data.get('alco', 0))                # 0, 1
        active = int(data.get('active', 1))            # 0, 1
        
        model_choice = data.get('model_choice', 'ensemble')  # 'ensemble' or 'scratch'
        
        # Computed features
        bmi = calculate_bmi(weight, height)
        bp_cat = categorize_bp(ap_hi, ap_lo)
        
        # Construct feature vector matching FEATURE_COLUMNS order
        input_data = pd.DataFrame([{
            'age_years': age_years,
            'gender': gender,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'bmi': bmi,
            'bp_category': bp_cat
        }])[FEATURE_COLUMNS]
        
        # Scale features
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data.values

        # Predict using selected model
        if model_choice == 'scratch' and scratch_model is not None:
            probs = scratch_model.predict_proba(input_scaled)[0]
            model_name_used = "Logistic Regression (Scratch Algorithm)"
        else:
            if best_model is not None:
                probs = best_model.predict_proba(input_scaled)[0]
                model_name_used = "Gradient Boosting (Ensemble Model)"
            elif scratch_model is not None:
                probs = scratch_model.predict_proba(input_scaled)[0]
                model_name_used = "Logistic Regression (Scratch Algorithm)"
            else:
                return jsonify({'error': 'Models not trained yet. Please run src/train.py first.'}), 500

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

        # Risk classification
        if risk_prob < 30.0:
            risk_level = "Low Risk"
            risk_class = "success"
            recommendation = "Your cardiovascular risk metrics are within normal ranges. Maintain a balanced diet, stay physically active, and schedule routine health check-ups."
        elif 30.0 <= risk_prob < 65.0:
            risk_level = "Moderate Risk"
            risk_class = "warning"
            recommendation = "Elevated risk detected. Consider monitoring blood pressure and cholesterol levels, reducing dietary sodium, and increasing physical activity."
        else:
            risk_level = "High Risk"
            risk_class = "danger"
            recommendation = "High risk of cardiovascular disease. We strongly recommend consulting a qualified physician or cardiologist for detailed medical screening."

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
            'model_used': model_name_used,
            'bmi': bmi,
            'bp_category_text': ['Normal', 'Elevated', 'Stage 1 Hypertension', 'Stage 2 Hypertension'][bp_cat]
        }

        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(result)
            
        return render_template('predict.html', result=result, input_values=data, metrics=metrics_data)

    except Exception as e:
        err_msg = str(e)
        if request.is_json:
            return jsonify({'success': False, 'error': err_msg}), 400
        return render_template('predict.html', error=err_msg, metrics=metrics_data)

@app.route('/model-info')
def model_info():
    """Model Comparison, ROC Curves, and Hyperparameters Page."""
    return render_template('model_info.html', metrics=metrics_data)

@app.route('/eda')
def eda():
    """Exploratory Data Analysis & Data Cleaning Stats Page."""
    return render_template('eda.html', metrics=metrics_data)

@app.route('/disclaimer')
def disclaimer():
    """Medical & Educational Disclaimer Page."""
    return render_template('disclaimer.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API Endpoint for programmatic prediction requests."""
    return predict()

if __name__ == '__main__':
    print("Starting CardioML Flask Application...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host='0.0.0.0', port=5000, debug=True)
