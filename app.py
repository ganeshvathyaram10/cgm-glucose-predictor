import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys

sys.path.insert(0, 'src')
from feature_engineering import get_feature_columns

st.set_page_config(
    page_title="CGM Glucose Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Blood Glucose Predictor")
st.markdown("Predicting blood glucose **30 minutes ahead** using a Random Forest model trained on CGM time-series data.")

st.divider()

@st.cache_resource
def load_model():
    model  = joblib.load('models/RandomForest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df_feat = pd.read_csv('data/processed/cgm_features.csv', parse_dates=['timestamp'])
    y_test  = np.load('reports/y_test.npy')
    y_pred  = np.load('reports/y_pred_best.npy')
    ts      = pd.to_datetime(pd.read_csv('reports/test_timestamps.csv').iloc[:, 0].values)
    imp_df  = pd.read_csv('reports/feature_importance.csv')
    return df_feat, y_test, y_pred, ts, imp_df

model, scaler = load_model()
df_feat, y_test, y_pred, ts, imp_df = load_data()
feature_cols = get_feature_columns(df_feat)

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Best Model", "Random Forest")
col2.metric("MAE", "2.39 mg/dL")
col3.metric("R² Score", "0.986")
col4.metric("Clarke Zone A", "100%")

st.divider()

st.subheader("Live 30-Minute Prediction")

sample_idx = st.slider(
    "Select a test sample",
    min_value=0,
    max_value=len(df_feat) - 1,
    value=100
)

sample_row   = df_feat.iloc[sample_idx]
sample_feats = sample_row[feature_cols].values.reshape(1, -1)
sample_scaled = scaler.transform(sample_feats)

predicted = model.predict(sample_scaled)[0]
current   = sample_row['glucose_mg_dl']
actual    = sample_row['target_glucose']
error     = abs(predicted - actual)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Glucose", f"{current:.1f} mg/dL")
col2.metric("Predicted (30 min)", f"{predicted:.1f} mg/dL")
col3.metric("Actual (30 min)", f"{actual:.1f} mg/dL")
col4.metric("Prediction Error", f"{error:.1f} mg/dL")

if predicted < 70:
    st.error("⚠️ Hypoglycemia predicted in 30 minutes — consider fast-acting carbohydrates")
elif predicted > 180:
    st.warning("⚠️ Hyperglycemia predicted in 30 minutes — consider a correction dose")
else:
    st.success("✅ Glucose predicted in target range (70–180 mg/dL)")

st.divider()

st.subheader("Actual vs Predicted — Test Set")

n = st.slider("Number of points to display", min_value=100, max_value=1000, value=300, step=50)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(ts[:n], y_test[:n], color="#60a5fa", lw=1.2, label="Actual")
ax.plot(ts[:n], y_pred[:n], color="#f472b6", lw=1.2, ls="--", label="Predicted")
ax.axhline(70,  color="#ef4444", lw=0.8, ls="--", alpha=0.6, label="Low threshold")
ax.axhline(180, color="#f59e0b", lw=0.8, ls="--", alpha=0.6, label="High threshold")
ax.set_ylabel("Glucose (mg/dL)")
ax.set_xlabel("Date")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close()

st.divider()

st.subheader("Top 20 Feature Importances")

top_n = st.slider("Number of features to display", min_value=5, max_value=20, value=10)

fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 1))
df_top = imp_df.head(top_n)
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(df_top)))[::-1]
ax.barh(df_top["feature"][::-1], df_top["importance"][::-1],
        color=colors, edgecolor="none", alpha=0.9)
ax.set_xlabel("Importance Score")
ax.grid(axis="x", alpha=0.3)

st.pyplot(fig)
plt.close()

st.divider()

st.markdown("Built with Random Forest · 48 engineered features · 90 days of CGM data")