"""
AirBnB Rating Prediction App
Model: v4
Workflow:
1. Upload features (X) → get predictions
2. Optional: upload true labels (Y) → evaluate model
"""

import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from src.features import prep_features


# Config

MODEL_PATH = "models/model_v5.pkl"
TRAINING_MEAN = 4.691  # baseline prediction

st.set_page_config(page_title="AirBnB Rating Predictor", page_icon="🏠")

st.title("🏠 AirBnB Rating Predictor")

# Load model

try:
    model = joblib.load(MODEL_PATH)
    FEATURE_ORDER = model.feature_names_in_
    st.success(f"✓ Model loaded ({len(FEATURE_ORDER)} features)")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# STEP 1 — Upload X and Predict

st.header("Upload data for prediction")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write(f"Uploaded **{len(df)} rows**")
    st.subheader("Preview")
    st.dataframe(df.head(10))
    st.map(df[["latitude", "longitude"]].dropna().head(1000), zoom=10, use_container_width=True, height=300)

    if st.button("Generate Predictions"):

        X = prep_features(df)

        # feature validation
        missing_cols = set(FEATURE_ORDER) - set(X.columns)
        if missing_cols:
            st.error(f"Missing required features: {list(missing_cols)}")
            st.stop()

        X = X[FEATURE_ORDER]

        with st.spinner("Running model..."):
            predictions = model.predict(X)

        # Save predictions
        st.session_state["predictions"] = predictions
        st.session_state["n_rows"] = len(predictions)

        st.success(f"{len(predictions)} predictions generated")


# STEP 1.5 — Show predictions (if exist)

if "predictions" in st.session_state:

    preds = st.session_state["predictions"]

    st.header("Prediction Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean", f"{preds.mean():.3f}")
    col2.metric("Min", f"{preds.min():.3f}")
    col3.metric("Max", f"{preds.max():.3f}")

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    st.dataframe(importance_df)

    st.bar_chart(
        importance_df.set_index("feature")
    )

    # download predictions
    output = pd.DataFrame({"prediction": preds})

    st.download_button(
        "Download Predictions",
        output.to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )


# STEP 2 — Evaluation
if "predictions" in st.session_state:

    st.header("Evaluate model performance")

    labels_file = st.file_uploader(
        "Upload true ratings CSV",
        type=["csv"],
        key="labels"
    )

    if labels_file:
        y_true = pd.read_csv(labels_file).iloc[:, 0].values
        preds = st.session_state["predictions"]

        if len(y_true) != len(preds):
            st.error(
                f"Label count ({len(y_true)}) "
                f"does not match predictions ({len(preds)})"
            )
            st.stop()

        # RMSE calculations
        model_rmse = np.sqrt(mean_squared_error(y_true, preds))
        dummy_preds = np.full(len(y_true), TRAINING_MEAN)
        dummy_rmse = np.sqrt(mean_squared_error(y_true, dummy_preds))

        improvement = ((dummy_rmse - model_rmse) / dummy_rmse) * 100

        st.subheader("Model Comparison")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model RMSE", f"{model_rmse:.4f}")
        col2.metric("Dummy RMSE", f"{dummy_rmse:.4f}")
        col3.metric("Improvement", f"{improvement:.1f}%")

        if improvement > 0:
            st.success(f"Model beats baseline by {improvement:.1f}%")
        else:
            st.error(f"Model is {-improvement:.1f}% worse than baseline")


# Reset button

if "predictions" in st.session_state:
    if st.button("Reset session"):
        st.session_state.clear()
        st.rerun()
