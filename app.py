"""
AirBnB Rating Prediction App - Final Version
Model: v3 with text keyword features (22 features)
"""

import joblib
import streamlit as st
import pandas as pd
import numpy as np
from src.features import prep_features

FEATURE_COLUMNS = joblib.load("models/feature_columns_v3.pkl")


st.set_page_config(page_title="AirBnB Rating Predictor", page_icon="🏠")

# Main app
st.title("AirBnB Rating Predictor")
st.write("Upload a CSV file to get rating predictions")

# Load model
try:
    model = joblib.load('models/model_v3.pkl')
    st.success("Model loaded")
except:
    st.error("Model not found! Run: python train.py --data-dir data")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded: **{len(df)} rows**")
    
    if st.button("Predict", type="primary"):
        X = prep_features(df)
        X = X[FEATURE_COLUMNS]
        predictions = model.predict(X)
        
        st.success(f"{len(predictions)} predictions generated")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{predictions.mean():.3f}")
        col2.metric("Min", f"{predictions.min():.3f}")
        col3.metric("Max", f"{predictions.max():.3f}")
        
        output = pd.DataFrame({'prediction': predictions})
        
        csv = output.to_csv(index=False)
        st.download_button("Download", csv, "predictions.csv", "text/csv")
