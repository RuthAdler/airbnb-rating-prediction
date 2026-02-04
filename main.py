"""
Main script to load data, preprocess, train the model, and save it.
"""
import pandas as pd

from src.data_loading import load_all_listings
from src.preprocessing import preprocess_data
from src.train import train_and_save
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

MODEL_PATH = "models/prod_pipeline.pkl"

def build_model():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

def load_training_data():
    datasets = load_all_listings("data")
    df = pd.concat(datasets.values(), ignore_index=True)
    return preprocess_data(df)

def main():
    X_train, X_test, y_train, y_test = load_training_data()

    model = build_model()
    model = train_and_save(model, X_train, y_train, path=MODEL_PATH)

    print(f"Saved final model to {MODEL_PATH}")

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Test MSE: {mse}")

if __name__ == "__main__":
    main()
