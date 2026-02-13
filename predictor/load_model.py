import joblib

def load_model(path: str):
    """Load a machine learning model from the specified file path."""

    return joblib.load(path)
