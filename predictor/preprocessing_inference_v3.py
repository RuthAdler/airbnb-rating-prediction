"""
Preprocessing for inference - VERSION 3 (GenAI submission)

Features:
- 15 simple features (same as v2)
- 7 text keyword features (no API needed)
- Ready for LLM features (optional)

Total: 22 features
"""

import pandas as pd
import numpy as np

# Feature columns
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


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess new data for prediction - GenAI version.
    
    Args:
        df: Raw AirBnB listings DataFrame
        
    Returns:
        DataFrame with 22 features ready for model
    """
    original_count = len(df)
    X = pd.DataFrame()
    
    # === Property features ===
    X['accommodates'] = df['accommodates'].fillna(2)
    X['bathrooms'] = df['bathrooms'].fillna(1)
    X['bedrooms'] = df['bedrooms'].fillna(1)
    X['beds'] = df['beds'].fillna(1)
    X['room_ratio'] = X['bedrooms'] / X['accommodates'].clip(lower=1)
    
    # === Host rates ===
    for col in ['host_response_rate', 'host_acceptance_rate']:
        X[col] = df[col].astype(str).str.replace('%', '').str.replace('N/A', '')
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(100) / 100
    
    # === Host features ===
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
    
    # === Booking features ===
    X['minimum_nights'] = np.log1p(df['minimum_nights'].fillna(1).clip(0, 365))
    X['instant_bookable'] = df['instant_bookable'].map({
        't': 1, 'f': 0, True: 1, False: 0
    }).fillna(0)
    
    # === Basic text features ===
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').apply(lambda x: len(str(x))).clip(0, 2000) / 2000
    X['has_host_about'] = df['host_about'].notna().astype(int)
    
    # === Text keyword features (GenAI-style without API) ===
    desc = df['description'].fillna('').str.lower()
    
    X['mentions_clean'] = desc.str.contains('clean|spotless|sanitize|hygien', regex=True).astype(int)
    X['mentions_luxury'] = desc.str.contains('luxury|luxurious|upscale|premium|elegant', regex=True).astype(int)
    X['mentions_view'] = desc.str.contains('view|views|skyline|ocean|beach|lake', regex=True).astype(int)
    X['mentions_location'] = desc.str.contains('walk|metro|subway|downtown|central|minute', regex=True).astype(int)
    X['mentions_modern'] = desc.str.contains('modern|new|renovated|updated|remodel', regex=True).astype(int)
    X['has_neighborhood'] = df['neighborhood_overview'].notna().astype(int)
    X['name_length'] = df['name'].fillna('').apply(len).clip(0, 100) / 100
    
    # === Final cleanup ===
    X = X[FEATURE_COLUMNS]
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.astype(float)
    
    assert len(X) == original_count, f"Row count changed: {original_count} -> {len(X)}"
    
    print(f"Preprocessed {len(X)} rows with {len(FEATURE_COLUMNS)} features")
    
    return X


# For testing
if __name__ == "__main__":
    print(f"Features ({len(FEATURE_COLUMNS)}):")
    for i, f in enumerate(FEATURE_COLUMNS, 1):
        print(f"  {i}. {f}")
