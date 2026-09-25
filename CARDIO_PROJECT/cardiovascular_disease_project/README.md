# Cardiovascular Disease Prediction (CardioML)
## Computer Engineering Department - ML Subject Project

This project implements an end-to-end Machine Learning pipeline and Flask web application for **Cardiovascular Disease Risk Prediction**, strictly adhering to the **Darshan University Standard Operating Procedure (SOP)**.

---

## 🌟 Key Highlights & SOP Compliance

| Week No. | Title | Implementation Details |
|---|---|---|
| **Week 1** | **Problem Definition & Dataset Exploration** | Kaggle Cardiovascular Dataset (`cardio_train.csv` - 70,000 records). Identifies target variable `cardio` (0: No CVD, 1: CVD Present). |
| **Week 2** | **Data Cleaning & Preprocessing** | Filters physiological outliers ($ap\_hi, ap\_lo$, height, weight). Converts age from days to years. Feature engineering: BMI ($kg/m^2$) & AHA Blood Pressure Categories. |
| **Week 3** | **Model Creation & Scratch Implementation** | **MANDATORY SOP CONSTRAINT**: Custom `ScratchLogisticRegression` built purely using NumPy vectorization & Gradient Descent. Comparative Scikit-Learn models (`RandomForest`, `GradientBoosting`, `DecisionTree`). |
| **Week 4** | **Model Evaluation** | Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC on holdout test data ($20\%$). |
| **Week 5** | **Advanced Model Training & Tuning** | 5-Fold Cross-Validation, feature importance evaluation, and model benchmarking. |
| **Week 6** | **Visualization of Metrics & Graphs** | Generates high-resolution performance plots: Comparative ROC Curves, Confusion Matrix, Feature Importances, and EDA distribution plots saved to `static/images/`. |
| **Week 7 - 10** | **Flask Web Setup, Front-end & Back-end Deployment** | Interactive web application (**CardioML**) featuring **Predict Now Form**, **Model Info & Benchmark**, **Data Insights / EDA**, and **Disclaimer**. |

---

## 🚀 Quick Start Guide

### 1. Run the Entire Project with One Single File:
```bash
python ML_PROJECT.py
```
This master script automatically:
1. Downloads `cardio_train.csv` dataset.
2. Performs data cleaning and feature engineering (70,000 raw $\rightarrow$ 68,528 clean records).
3. Trains the **Scratch Logistic Regression** model and ML library models (`Gradient Boosting`, `Random Forest`, `Decision Tree`).
4. Generates ROC curves, Confusion Matrix, and EDA plots in `static/images/`.
5. Saves trained model artifacts in `models/`.
6. Launches the **CardioML Web Server** at `http://127.0.0.1:5000`.

---

## 📁 Project Directory Structure

```text
cardiovascular_disease_project/
├── ML_PROJECT.py                  # Master executable containing ALL pipeline steps & Flask server
├── app.py                         # Standalone Flask app script
├── data/
│   └── cardio_train.csv           # Kaggle 70,000 record dataset
├── src/
│   ├── preprocessing.py           # Data cleaning & feature engineering
│   ├── scratch_model.py           # Custom Scratch Logistic Regression (NumPy)
│   └── train.py                   # Model training & visualization script
├── models/
│   ├── best_model.pkl             # Trained Gradient Boosting pipeline
│   ├── scratch_model.pkl          # Saved Scratch Logistic Regression model
│   ├── scaler.pkl                 # StandardScaler instance
│   └── metrics.json               # Full evaluation metrics & feature importances
├── static/
│   ├── css/style.css              # Custom CardioML UI styling
│   ├── js/main.js                 # Dynamic form interaction script
│   └── images/                    # ROC Curve, Confusion Matrix, Feature Importance & EDA plots
├── templates/
│   ├── base.html                  # Main layout template with CardioML navigation
│   ├── index.html                 # Landing / Hero page (SOP Page 7 screenshot)
│   ├── predict.html               # Risk assessment form (SOP Page 8 screenshot)
│   ├── model_info.html            # Model comparison table & metrics (SOP Page 8 screenshot)
│   ├── eda.html                   # Data insights & healthy targets (SOP Page 9 screenshot)
│   └── disclaimer.html            # Academic disclaimer page
└── README.md                      # Project documentation
```

---

## 📊 Model Benchmark Results

| Model / Algorithm | Accuracy (%) | F1 Score (%) | ROC AUC (%) | SOP Compliance |
|---|:---:|:---:|:---:|:---:|
| **Gradient Boosting Classifier** | **73.4%** | **71.7%** | **79.7%** | Scikit-Learn Library (Best Model) |
| **Random Forest Classifier** | 72.8% | 71.1% | 78.9% | Scikit-Learn Library |
| **Scratch Logistic Regression** | **72.1%** | **70.4%** | **77.8%** | **Mandatory Scratch Algorithm** |
| **Sklearn Logistic Regression** | 72.2% | 70.5% | 77.9% | Scikit-Learn Library |
| **Decision Tree Classifier** | 72.0% | 70.2% | 77.2% | Scikit-Learn Library |

---

## 🔗 Live Application Access
- **Web Interface**: `http://127.0.0.1:5000`
- **Predict Route**: `http://127.0.0.1:5000/predict`
- **Model Info Route**: `http://127.0.0.1:5000/model-info`
- **EDA Route**: `http://127.0.0.1:5000/eda`
- **API Endpoint**: `POST http://127.0.0.1:5000/api/predict`
