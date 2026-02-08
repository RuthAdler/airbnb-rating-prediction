import joblib
import os
import streamlit as st
import pandas as pd

from predictor.config import MODEL_PATH, SCALER_PATH
from predictor.preprocessing_inference import preprocess_for_inference

"""
AirBnB Rating Prediction App
"""


# Page setup
@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(current_dir, MODEL_PATH)
    scaler_path = os.path.join(current_dir, SCALER_PATH)

    model = joblib.load(model_path)

    # Optional: Load scaler if it exists
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    return model, scaler


# Main app
st.title("AirBnB Rating Predictor")
st.write("Upload a CSV file with AirBnB listings to get rating predictions.")

# Try to load the model
try:
    model, scaler = load_resources()
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
    if scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values  # Convert to numpy array if not already

    predictions = model.predict(X)

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
