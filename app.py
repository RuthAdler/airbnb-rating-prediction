import streamlit as st
import pandas as pd
import joblib
from predictor.pipeline import predict
from predictor.config import DEFAULT_MODEL_PATH
from src.preprocessing_inference import preprocess_for_inference


#TODO: Build a Streamlit app

"""
AirBnB Rating Prediction App
"""

# Page setup
st.set_page_config(page_title="AirBnB Rating Predictor", layout="centered")

# Load the model and scaler (this only runs once)
@st.cache_resource
def load_resources():
    model = joblib.load('models/scaler.pkl')
    scaler = joblib.load('models/best_model.pkl')
    return model, scaler

# Main app
st.title("AirBnB Rating Predictor")
st.write("Upload a CSV file with AirBnB listings to get rating predictions.")

# Try to load the model
try:
    model, scaler = load_model()
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Read the file
    try:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded {len(df)} rows")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
    
    # Preprocess
    try:
        X = preprocess_for_inference(df)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    # Create output dataframe
    output = pd.DataFrame({'prediction': predictions})
    
    # Show first few predictions
    st.write("Preview (first 10 rows):")
    st.dataframe(output.head(10))
    
    # Download button
    csv = output.to_csv(index=False)
    st.download_button(
        label="Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
    
    st.write(f"Done! Generated {len(predictions)} predictions.")