# Blood Glucose Prediction — 30-Minute CGM Forecasting

> Live demo: https://cgm-glucose-predictor.streamlit.app

Predicting future blood glucose levels 30 minutes ahead using continuous glucose monitor (CGM) data and ensemble machine learning.

## Results

| Model | MAE (mg/dL) | RMSE (mg/dL) | R² |
|---|---|---|---|
| Random Forest | 2.39 | 3.00 | 0.986 |
| XGBoost | 2.43 | 3.05 | 0.986 |
| Ridge | 3.49 | 4.37 | 0.970 |
| Linear Regression | 3.49 | 4.37 | 0.970 |

## Quick Start
git clone https://github.com/ganeshvathyaram10/cgm-glucose-predictor
cd cgm-glucose-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/data_generator.py
python3 src/data_cleaning.py
python3 src/feature_engineering.py
python3 src/model_training.py
python3 src/visualizations.py
streamlit run app.py

## Resume Bullet Points
- Built end-to-end ML pipeline predicting blood glucose 30 minutes ahead with 2.39 mg/dL MAE and R² = 0.986
- Engineered 48 features from raw sensor data including lag variables, rolling statistics, and rate-of-change derivatives
- Compared 4 ML models using chronological train/test split to prevent data leakage in time-series forecasting
- Deployed interactive prediction dashboard on Streamlit Cloud with live 30-minute glucose forecasting
