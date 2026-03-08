from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error
from src.features import prep_features

# Config
BASE = Path(__file__).parent / ".."
X_PATH = BASE / "data" / "test" / "listings Sydney X.csv"
Y_PATH = BASE / "data" / "test" / "listings Sydney y.csv"
MODEL_PATH = BASE / "models" / "model_v5-1.pkl"

def main():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    df_x = pd.read_csv(X_PATH)
    df_y = pd.read_csv(Y_PATH)

    #
    X = prep_features(df_x)[model.feature_names_in_]

    y_true = df_y['review_scores_rating'].values if 'review_scores_rating' in df_y.columns else df_y.iloc[:, 0].values

    preds = model.predict(X)
    rmse = root_mean_squared_error(y_true, preds)

    print("\n--- Results ---")
    print(f"Total predictions: {len(preds)}")
    print(f"Predicted Mean: {preds.mean():.4f}")
    print(f"Test RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()