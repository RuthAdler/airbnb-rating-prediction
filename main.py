"""
Main script to load data, preprocess, train the model, and save it.
"""
import pandas as pd

from src.data_loading import load_all_listings
from src.preprocessing import preprocess_data
from src.train import train_and_save
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

MODEL_PATH = "models/prod_pipeline.pkl"


def build_pipeline(model, scaler=None):
    steps = []
    if scaler:
        steps.append(('scaler', scaler))
    steps.append(('model', model))
    return Pipeline(steps)


def load_training_data():
    datasets = load_all_listings("data")
    df = pd.concat(datasets.values(), ignore_index=True)
    return preprocess_data(df)


def main(model=None, scaler=None):
    X_train, X_test, y_train, y_test = load_training_data()

    pipline = build_pipeline(model, scaler)
    print("Training production pipeline...")

    pipeline = train_and_save(pipline, X_train, y_train, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Test MSE: {mse}")


if __name__ == "__main__":
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    scaler = StandardScaler()

    main(model=rf, scaler=scaler)
