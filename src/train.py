"""
Train the rating prediction model.

Usage:
    python -m src.train --data-dir data --output-dir models

Creates:
    - models/model_<version>.pkl
    - models/feature_columns_<version>.pkl
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib
from xgboost import XGBRegressor


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import prep_features, FEATURE_COLUMNS


def load_training_data(data_dir: str) -> pd.DataFrame:
    """Load and combine training datasets."""
    data_path = Path(data_dir)
    datasets = []
    for csv_file in data_path.glob('listings*.csv'):
        if 'TEST' in csv_file.name.upper():
            continue
        print(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        datasets.append(df)
        print(f"  {len(df)} rows")

    combined = pd.concat(datasets, ignore_index=True)
    print(f"Total: {len(combined)} rows")
    return combined


def main(data_dir: str, output_dir: str = 'models', model_version: str = 'v0'):
    """Train and save the model."""

    Path(output_dir).mkdir(exist_ok=True)

    # Load data
    print("\n=== Loading Data ===")
    df = load_training_data(data_dir)
    df = df.dropna(subset=['review_scores_rating'])
    print(f"After dropping missing targets: {len(df)} rows")

    # Prepare features
    print("\n=== Preparing Features ===")
    X = prep_features(df)
    y = df['review_scores_rating'].values
    print(f"Features: {len(FEATURE_COLUMNS)}")

    # Try multiple models
    print("\n=== Training Models ===")

    models = {
        'Ridge': Ridge(alpha=10),

        'Scaler + Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10))
        ]),

        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        ),

        'XGBoost': XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:squarederror"
        )
    }

    best_model = None
    best_rmse = float('inf')
    best_name = None

    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
        cv_rmse = -cv_scores.mean()
        print(f"{name}: CV RMSE = {cv_rmse:.4f} (+/- {cv_scores.std():.4f})")

        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_model = model
            best_name = name

    print(f"\nBest: {best_name} ({best_rmse:.4f})")

    # Train final model
    print("\n=== Training Final Model ===")
    best_model.fit(X, y)

    # Save
    print("\n=== Saving ===")
    joblib.dump(best_model, f'{output_dir}/model_{model_version}.pkl')
    joblib.dump(FEATURE_COLUMNS, f'{output_dir}/feature_columns_{model_version}.pkl')

    print(f"Saved: {output_dir}/model_{model_version}.pkl")
    print(f"Saved: {output_dir}/feature_columns_{model_version}.pkl")

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='models')
    args = parser.parse_args()

    MODEL_VERSION = 'v5'

    main(args.data_dir, args.output_dir, model_version=MODEL_VERSION)
