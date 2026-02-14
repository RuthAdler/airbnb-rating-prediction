"""
AirBnB Rating Prediction App - Final Version
Model: v3 with text keyword features (22 features)
"""

import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="AirBnB Rating Predictor", page_icon="🏠")

# Training mean (calculated from LA + NYC data)
TRAINING_MEAN = 4.691  # Dummy baseline predicts this

# Exact feature order from training
FEATURE_ORDER = [
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'room_ratio',
    'host_response_rate', 'host_acceptance_rate', 'is_superhost',
    'host_days_log', 'minimum_nights', 'instant_bookable', 'has_description',
    'desc_length', 'has_host_about', 'response_speed', 'mentions_clean',
    'mentions_luxury', 'mentions_view', 'mentions_location', 'mentions_modern',
    'has_neighborhood', 'name_length'
]


def preprocess(df):
    """Preprocess data - 22 features in exact order."""
    X = pd.DataFrame()
    
    # Property features
    X['accommodates'] = df['accommodates'].fillna(2)
    X['bathrooms'] = df['bathrooms'].fillna(1)
    X['bedrooms'] = df['bedrooms'].fillna(1)
    X['beds'] = df['beds'].fillna(1)
    X['room_ratio'] = X['bedrooms'] / X['accommodates'].clip(lower=1)
    
    # Host rates
    for col in ['host_response_rate', 'host_acceptance_rate']:
        X[col] = df[col].astype(str).str.replace('%', '').str.replace('N/A', '')
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(100) / 100
    
    # Host features
    X['is_superhost'] = df['host_is_superhost'].map({'t':1,'f':0,True:1,False:0}).fillna(0)
    
    df2 = df.copy()
    df2['host_since'] = pd.to_datetime(df2['host_since'], errors='coerce', dayfirst=True)
    df2['last_scraped'] = pd.to_datetime(df2['last_scraped'], errors='coerce', dayfirst=True)
    X['host_days_log'] = np.log1p((df2['last_scraped']-df2['host_since']).dt.days.fillna(0).clip(lower=0))
    
    X['response_speed'] = df['host_response_time'].map({
        'within an hour':1,'within a few hours':0.75,'within a day':0.5,'a few days or more':0.25
    }).fillna(0.5)
    
    # Booking features
    X['minimum_nights'] = np.log1p(df['minimum_nights'].fillna(1).clip(0,365))
    X['instant_bookable'] = df['instant_bookable'].map({'t':1,'f':0,True:1,False:0}).fillna(0)
    
    # Text features
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').apply(lambda x: len(str(x))).clip(0,2000)/2000
    X['has_host_about'] = df['host_about'].notna().astype(int)
    
    # Text keyword features
    desc = df['description'].fillna('').str.lower()
    X['mentions_clean'] = desc.str.contains('clean|spotless|sanitize|hygien', regex=True).astype(int)
    X['mentions_luxury'] = desc.str.contains('luxury|luxurious|upscale|premium|elegant', regex=True).astype(int)
    X['mentions_view'] = desc.str.contains('view|views|skyline|ocean|beach|lake', regex=True).astype(int)
    X['mentions_location'] = desc.str.contains('walk|metro|subway|downtown|central|minute', regex=True).astype(int)
    X['mentions_modern'] = desc.str.contains('modern|new|renovated|updated|remodel', regex=True).astype(int)
    X['has_neighborhood'] = df['neighborhood_overview'].notna().astype(int)
    X['name_length'] = df['name'].fillna('').apply(len).clip(0, 100) / 100
    
    # IMPORTANT: Return features in exact order
    return X[FEATURE_ORDER].fillna(0)


# Main app
st.title("🏠 AirBnB Rating Predictor")
st.write("Upload a CSV file to get rating predictions")

# Load model
try:
    model = joblib.load('models/model_v3.pkl')
    st.success("✓ Model loaded (22 features)")
except:
    st.error("Model not found! Run: python train.py --data-dir data")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload features CSV (X)", type=['csv'])
labels_file = st.file_uploader("Upload labels CSV (Y) - optional, for RMSE comparison", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded: **{len(df)} rows**")
    
    if st.button("🚀 Predict", type="primary"):
        X = preprocess(df)
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
