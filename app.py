"""
AirBnB Rating Prediction App - Final Version
Model: v3 with text keyword features (22 features)
"""

import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from src.features import prep_features

MODEL_PATH = 'models/model_v3.pkl'
st.set_page_config(page_title="AirBnB Rating Predictor", page_icon="🏠")

# Training mean (calculated from LA + NYC data)
TRAINING_MEAN = 4.691  # Dummy baseline predicts this

# Main app
st.title("🏠 AirBnB Rating Predictor")
st.write("Upload a CSV file to get rating predictions")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    FEATURE_ORDER = model.feature_names_in_
    st.success(f"✓ Model loaded ({len(FEATURE_ORDER)} features)")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload features CSV (X)", type=['csv'])
labels_file = st.file_uploader("Upload labels CSV (Y) - optional, for RMSE comparison", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded: **{len(df)} rows**")

    if st.button("🚀 Predict", type="primary"):
        X = prep_features(df)

        missing_cols = set(FEATURE_ORDER) - set(X.columns)
        if missing_cols:
            st.error(f"Missing required features: {list(missing_cols)}")
            st.stop()

        X = X[FEATURE_ORDER]

        predictions = model.predict(X)

        st.success(f"✅ {len(predictions)} predictions generated")

        # Prediction stats
        st.subheader("📊 Prediction Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{predictions.mean():.3f}")
        col2.metric("Min", f"{predictions.min():.3f}")
        col3.metric("Max", f"{predictions.max():.3f}")

        # If labels provided, show RMSE comparison
        if labels_file:
            y_true = pd.read_csv(labels_file).iloc[:, 0].values

            # Model RMSE
            model_rmse = np.sqrt(mean_squared_error(y_true, predictions))

            # Dummy RMSE (predicts training mean)
            dummy_preds = np.full(len(y_true), TRAINING_MEAN)
            dummy_rmse = np.sqrt(mean_squared_error(y_true, dummy_preds))

            # Improvement
            improvement = ((dummy_rmse - model_rmse) / dummy_rmse) * 100

            st.subheader("📈 Model Comparison")
            col1, col2, col3 = st.columns(3)
            col1.metric("Our Model RMSE", f"{model_rmse:.4f}")
            col2.metric("Dummy RMSE", f"{dummy_rmse:.4f}")
            col3.metric("Improvement", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")

            if model_rmse < dummy_rmse:
                st.success(f"✅ Our model beats the baseline by {improvement:.1f}%!")
            else:
                st.error(f"❌ Our model is {-improvement:.1f}% worse than baseline")

        # Download
        output = pd.DataFrame({'prediction': predictions})
        csv = output.to_csv(index=False)
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
