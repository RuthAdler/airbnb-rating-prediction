"""
Train the simple model that generalizes across cities.

Usage:
    python train_simple.py --data-dir data

This creates:
    - models/best_model.pkl (the model)
    - models/feature_columns.pkl (feature names)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib


FEATURE_COLUMNS = [
    'accommodates',
    'bathrooms', 
    'bedrooms',
    'beds',
    'room_ratio',
    'host_response_rate',
    'host_acceptance_rate',
    'is_superhost',
    'host_days_log',
    'minimum_nights',
    'instant_bookable',
    'has_description',
    'desc_length',
    'has_host_about',
    'response_speed'
]


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from raw data."""
    X = pd.DataFrame()
    
    # Property features
    X['accommodates'] = df['accommodates'].fillna(2)
    X['bathrooms'] = df['bathrooms'].fillna(1)
    X['bedrooms'] = df['bedrooms'].fillna(1)
    X['beds'] = df['beds'].fillna(1)
    X['room_ratio'] = X['bedrooms'] / X['accommodates'].clip(lower=1)
    
    # Host rates
    for col in ['host_response_rate', 'host_acceptance_rate']:
        X[col] = df[col].astype(str).str.replace('%', '').str.replace('N/A', '')
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(100) / 100
    
    # Host features
    X['is_superhost'] = df['host_is_superhost'].map({
        't': 1, 'f': 0, True: 1, False: 0
    }).fillna(0)
    
    df_copy = df.copy()
    df_copy['host_since'] = pd.to_datetime(df_copy['host_since'], errors='coerce', dayfirst=True)
    df_copy['last_scraped'] = pd.to_datetime(df_copy['last_scraped'], errors='coerce', dayfirst=True)
    host_days = (df_copy['last_scraped'] - df_copy['host_since']).dt.days.fillna(0).clip(lower=0)
    X['host_days_log'] = np.log1p(host_days)
    
    resp_map = {
        'within an hour': 1.0, 
        'within a few hours': 0.75, 
        'within a day': 0.5, 
        'a few days or more': 0.25
    }
    X['response_speed'] = df['host_response_time'].map(resp_map).fillna(0.5)
    
    # Booking features
    X['minimum_nights'] = np.log1p(df['minimum_nights'].fillna(1).clip(0, 365))
    X['instant_bookable'] = df['instant_bookable'].map({
        't': 1, 'f': 0, True: 1, False: 0
    }).fillna(0)
    
    # Text features
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').apply(lambda x: len(str(x))).clip(0, 2000) / 2000
    X['has_host_about'] = df['host_about'].notna().astype(int)
    
    return X[FEATURE_COLUMNS].fillna(0)


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


def main(data_dir: str, output_dir: str = 'models'):
    """Train and save the model."""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    print("\n=== Loading Data ===")
    df = load_training_data(data_dir)
    
    # Remove rows without target
    df = df.dropna(subset=['review_scores_rating'])
    print(f"After dropping missing targets: {len(df)} rows")
    
    # Prepare features
    print("\n=== Preparing Features ===")
    X = prep_features(df)
    y = df['review_scores_rating'].values
    
    print(f"Features: {list(X.columns)}")
    print(f"X shape: {X.shape}")
    
    # Cross-validation
    print("\n=== Cross-Validation ===")
    model = Ridge(alpha=10)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()
    print(f"CV RMSE: {cv_rmse:.4f} (+/- {cv_scores.std():.4f})")
    
    # Train final model
    print("\n=== Training Final Model ===")
    model.fit(X, y)
    
    # Save
    print("\n=== Saving ===")
    joblib.dump(model, f'{output_dir}/best_model.pkl')
    joblib.dump(FEATURE_COLUMNS, f'{output_dir}/feature_columns.pkl')
    
    print(f"Saved: {output_dir}/best_model.pkl")
    print(f"Saved: {output_dir}/feature_columns.pkl")
    
    # Show feature importance
    print("\n=== Feature Coefficients ===")
    for name, coef in sorted(zip(FEATURE_COLUMNS, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {coef:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Directory with training CSVs')
    parser.add_argument('--output-dir', default='models', help='Directory to save model')
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
