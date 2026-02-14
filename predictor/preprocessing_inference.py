"""
Preprocessing for inference (new data) - UPDATED VERSION

This version uses simple, universal features that generalize across cities.
Tested on Chicago data - beats baseline by ~6%.
"""

import pandas as pd
import numpy as np

# These are the 15 features the model expects
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


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess new data for prediction.
    
    Uses simple, universal features that work across different cities.
    Never drops rows - every input row gets a prediction.
    
    Args:
        df: Raw AirBnB listings DataFrame
        
    Returns:
        DataFrame with 15 features ready for model
    """
    original_count = len(df)
    X = pd.DataFrame()
    
    # === Property features ===
    X['accommodates'] = df['accommodates'].fillna(2)
    X['bathrooms'] = df['bathrooms'].fillna(1)
    X['bedrooms'] = df['bedrooms'].fillna(1)
    X['beds'] = df['beds'].fillna(1)
    X['room_ratio'] = X['bedrooms'] / X['accommodates'].clip(lower=1)
    
    # === Host rates (clean % signs) ===
    for col in ['host_response_rate', 'host_acceptance_rate']:
        X[col] = df[col].astype(str).str.replace('%', '').str.replace('N/A', '')
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(100) / 100
    
    # === Host features ===
    X['is_superhost'] = df['host_is_superhost'].map({
        't': 1, 'f': 0, True: 1, False: 0
    }).fillna(0)
    
    # Host experience (log of days since signup)
    df_copy = df.copy()
    df_copy['host_since'] = pd.to_datetime(df_copy['host_since'], errors='coerce', dayfirst=True)
    df_copy['last_scraped'] = pd.to_datetime(df_copy['last_scraped'], errors='coerce', dayfirst=True)
    host_days = (df_copy['last_scraped'] - df_copy['host_since']).dt.days.fillna(0).clip(lower=0)
    X['host_days_log'] = np.log1p(host_days)
    
    # Response speed (normalized score)
    resp_map = {
        'within an hour': 1.0, 
        'within a few hours': 0.75, 
        'within a day': 0.5, 
        'a few days or more': 0.25
    }
    X['response_speed'] = df['host_response_time'].map(resp_map).fillna(0.5)
    
    # === Booking features ===
    X['minimum_nights'] = np.log1p(df['minimum_nights'].fillna(1).clip(0, 365))
    X['instant_bookable'] = df['instant_bookable'].map({
        't': 1, 'f': 0, True: 1, False: 0
    }).fillna(0)
    
    # === Text features (just presence/length, not content) ===
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').apply(lambda x: len(str(x))).clip(0, 2000) / 2000
    X['has_host_about'] = df['host_about'].notna().astype(int)
    
    # === Final cleanup ===
    X = X[FEATURE_COLUMNS]  # Ensure correct column order
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.astype(float)
    
    # Verify no rows lost
    assert len(X) == original_count, f"Row count changed: {original_count} -> {len(X)}"
    
    print(f"Preprocessed {len(X)} rows with {len(FEATURE_COLUMNS)} features")
    
    return X


# For testing
if __name__ == "__main__":
    print("Testing preprocessing_inference.py")
    print(f"Features: {FEATURE_COLUMNS}")
    
    test_data = pd.DataFrame({
        'accommodates': [2, 4, 3],
        'bathrooms': [1, None, 2],
        'bedrooms': [1, 2, None],
        'beds': [1, None, 2],
        'host_is_superhost': ['t', 'f', None],
        'host_response_rate': ['100%', '90%', None],
        'host_acceptance_rate': ['95%', None, '80%'],
        'host_since': ['2020-01-01', '2018-06-15', None],
        'last_scraped': ['2024-01-01', '2024-01-01', '2024-01-01'],
        'host_response_time': ['within an hour', None, 'within a day'],
        'minimum_nights': [1, 30, None],
        'instant_bookable': ['t', 'f', 't'],
        'description': ['Nice place', None, 'A' * 500],
        'host_about': ['I love hosting', None, None],
    })
    
    result = preprocess_for_inference(test_data)
    print(f"\nOutput shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print("\nFirst row:")
    print(result.iloc[0])
