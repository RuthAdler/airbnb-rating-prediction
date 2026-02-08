import pandas as pd
from src.preprocessing_inference import preprocess_for_inference
import joblib
import streamlit as st
from predictor.config import DEFAULT_MODEL_PATH


@st.cache_resource
def load_model(path: str = DEFAULT_MODEL_PATH) -> object:
    """Load a machine learning model from the specified file path."""

    return joblib.load(path)

def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the raw input data for inference."""
    return preprocess_for_inference(df_raw)

def predict(df_processed: pd.DataFrame, model: object) -> pd.Series:
    """Make predictions using the loaded model on the preprocessed data."""
    predictions = model.predict(df_processed)
    return predictions
