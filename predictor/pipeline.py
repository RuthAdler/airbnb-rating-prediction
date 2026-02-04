import pandas as pd
from .config import DEFAULT_MODEL_PATH
from .load_model import load_model
from src.preprocessing_inference import preprocess_for_inference

_model = None


def predict(df_raw: pd.DataFrame):
    """Predict using the trained model. This function is designed to be called multiple times, but the model will only be loaded once."""

    global _model

    if _model is None:
        _model = load_model(DEFAULT_MODEL_PATH)

    X = preprocess_for_inference(df_raw)

    return _model.predict(X)
