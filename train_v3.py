"""
Train model v3 - with text keyword features (GenAI submission)

Usage:
    python train_v3.py --data-dir data

Creates:
    - models/model_v3.pkl
    - models/feature_columns_v3.pkl
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib


FEATURE_COLUMNS = [
    # Original 15
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
    'response_speed',
    # Text keywords (7)
    'mentions_clean',
    'mentions_luxury',
    'mentions_view',
    'mentions_location',
    'mentions_modern',
    'has_neighborhood',
    'name_length',
]


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features from raw data."""
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
    
    # Basic text features
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').apply(lambda x: len(str(x))).clip(0, 2000) / 2000
    X['has_host_about'] = df['host_about'].notna().astype(int)
    
    # Text keyword features
    desc = df['description'].fillna('').str.lower()
    X['mentions_clean'] = desc.str.contains('clean|spotless|sanitize|hygien', regex=True).astype(int)
    X['mentions_luxury'] = desc.str.contains('luxury|luxurious|upscale|premium|elegant', regex=True).astype(int)
    X['mentions_view'] = desc.str.contains('view|views|skyline|ocean|beach|lake', regex=True).astype(int)
    X['mentions_location'] = desc.str.contains('walk|metro|subway|downtown|central|minute', regex=True).astype(int)
    X['mentions_modern'] = desc.str.contains('modern|new|renovated|updated|remodel', regex=True).astype(int)
    X['has_neighborhood'] = df['neighborhood_overview'].notna().astype(int)
    X['name_length'] = df['name'].fillna('').apply(len).clip(0, 100) / 100
    
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
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
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
    joblib.dump(best_model, f'{output_dir}/model_v3.pkl')
    joblib.dump(FEATURE_COLUMNS, f'{output_dir}/feature_columns_v3.pkl')
    
    print(f"Saved: {output_dir}/model_v3.pkl")
    print(f"Saved: {output_dir}/feature_columns_v3.pkl")
    
    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='models')
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
