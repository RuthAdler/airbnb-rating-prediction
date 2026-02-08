"""
Preprocessing for inference (new data).

This is different from training preprocessing because:
- We don't have the target column (review_scores_rating)
- We can NOT drop any rows - every row needs a prediction
- We use fixed values instead of calculating from data
"""

import pandas as pd
import numpy as np
from src.geo_processing import add_distance_to_center

# Fixed values from training data - use these for filling missing values
IMPUTATION_VALUES = {
    'host_response_rate_median': 1.0,
    'host_acceptance_rate_median': 0.99,
    'bathrooms_default': 1.0,
    'bedrooms_default': 1.0,
    'beds_default': 2.0,
    'log_price_default': 4.8,
    'latitude_default': 34.05,   # roughly LA
    'longitude_default': -118.25,
}

# These are the EXACT 25 features the model was trained on
FEATURE_COLUMNS = [
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'distance_to_center',           # Calculated below
    'estimated_occupancy_l365d',
    'has_host_about',
    'has_neighborhood_overview',
    'host_acceptance_rate',
    'host_is_superhost',
    'host_response_rate',
    'host_response_time_coded',
    'host_tenure_days',             # Calculated below
    'instant_bookable',
    'log_host_listings_count',
    'log_host_total_listings_count',
    'log_price',
    'maximum_maximum_nights',
    'maximum_minimum_nights',
    'maximum_nights',
    'maximum_nights_avg_ntm',
    'minimum_maximum_nights',
    'minimum_minimum_nights',
    'minimum_nights',
    'minimum_nights_avg_ntm'
]


def convert_data_types(df):
    """Convert columns to appropriate data types."""
    df = df.copy()
    
    # Datetime columns (needed for some feature engineering)
    date_cols = ['last_scraped', 'host_since', 'first_review', 'last_review']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Numeric columns (remove $, %, commas)
    numeric_cols = ['host_response_rate', 'host_acceptance_rate', 'price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace(r'\s*[$%,]\s*', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Boolean columns
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'instant_bookable']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].replace({'t': True, 'f': False})
            df[col] = df[col].fillna(False)
    
    return df


def create_features(df):
    """Create engineered features."""
    df = df.copy()
    
    # 1. Host Response Time
    if 'host_response_time' in df.columns:
        mapping = {
            'within an hour': 1, 
            'within a few hours': 2, 
            'within a day': 3, 
            'a few days or more': 4
        }
        df['host_response_time_coded'] = df['host_response_time'].map(mapping).fillna(0)
    else:
        df['host_response_time_coded'] = 0
    
    # 2. Booleans
    df['has_neighborhood_overview'] = df['neighborhood_overview'].notna() if 'neighborhood_overview' in df.columns else False
    df['has_host_about'] = df['host_about'].notna() if 'host_about' in df.columns else False
    
    # 3. Log Price
    if 'price' in df.columns:
        df['log_price'] = np.log(df['price'].clip(lower=1))
        df['log_price'] = df['log_price'].fillna(IMPUTATION_VALUES['log_price_default'])
    else:
        df['log_price'] = IMPUTATION_VALUES['log_price_default']
    
    # 4. Log Host Listings
    for col in ['host_total_listings_count', 'host_listings_count']:
        new_col = f"log_{col}"
        if col in df.columns:
            df[new_col] = np.log(df[col].fillna(1) + 1)
        else:
            df[new_col] = 0

    # 5. Occupancy
    if 'availability_365' in df.columns:
        df['estimated_occupancy_l365d'] = (365 - df['availability_365']).fillna(0)
    else:
        df['estimated_occupancy_l365d'] = 0
    
    # 6. Host Tenure Days
    if 'host_since' in df.columns:
        ref_date = df['last_scraped'] if 'last_scraped' in df.columns else pd.Timestamp.now()
        df['host_tenure_days'] = (ref_date - df['host_since']).dt.days.fillna(0)
    else:
        df['host_tenure_days'] = 0

    # 7. Distance to Center (Matches logic in geo_preprocessing.py)
    # We use fixed coordinates because we don't have the training data at inference time
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Match keys in IMPUTATION_VALUES
        center = (IMPUTATION_VALUES['longitude_default'], IMPUTATION_VALUES['latitude_default'])
            
        # Call the imported function
        df = add_distance_to_center(df, center)
        df['distance_to_center'] = df['distance_to_center'].fillna(0)
    else:
        df['distance_to_center'] = 0

    return df


def handle_missing_values(df):
    """Fill missing values with fixed defaults. Never drop rows."""
    df = df.copy()
    
    # Night-related columns: fill with base values
    if 'minimum_nights' in df.columns:
        df['minimum_minimum_nights'] = df.get('minimum_minimum_nights', df['minimum_nights']).fillna(df['minimum_nights'])
        df['maximum_minimum_nights'] = df.get('maximum_minimum_nights', df['minimum_nights']).fillna(df['minimum_nights'])
    
    if 'maximum_nights' in df.columns:
        df['minimum_maximum_nights'] = df.get('minimum_maximum_nights', df['maximum_nights']).fillna(df['maximum_nights'])
        df['maximum_maximum_nights'] = df.get('maximum_maximum_nights', df['maximum_nights']).fillna(df['maximum_nights'])
    
    # Fill columns with defaults
    defaults = {
        'host_is_superhost': False,
        'instant_bookable': False,
        'host_response_rate': IMPUTATION_VALUES['host_response_rate_median'],
        'host_acceptance_rate': IMPUTATION_VALUES['host_acceptance_rate_median'],
        'bathrooms': IMPUTATION_VALUES['bathrooms_default'],
        'bedrooms': IMPUTATION_VALUES['bedrooms_default'],
        'beds': IMPUTATION_VALUES['beds_default'],
        'accommodates': 2,
        'minimum_nights': 1,
        'maximum_nights': 365,
        'minimum_nights_avg_ntm': 1,
        'maximum_nights_avg_ntm': 365,
        'minimum_minimum_nights': 1,
        'maximum_minimum_nights': 1,
        'minimum_maximum_nights': 365,
        'maximum_maximum_nights': 365,
        'latitude': IMPUTATION_VALUES['latitude_default'],
        'longitude': IMPUTATION_VALUES['longitude_default'],
        'estimated_occupancy_l365d': 0,
        'has_host_about': False,
        'has_neighborhood_overview': False,
        'host_response_time_coded': 0,
        'log_price': IMPUTATION_VALUES['log_price_default'],
        'log_host_listings_count': 0,
        'log_host_total_listings_count': 0,
        'description_length_words': 0,
        'description_length_chars': 0,
    }
    
    for col, default_val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
        else:
            df[col] = default_val
    
    return df


def select_features(df):
    """Select only the 25 features the model expects."""
    
    # Make sure all expected features exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the expected features in the correct order
    X = df[FEATURE_COLUMNS].copy()
    
    return X


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function: preprocess new data for prediction.
    
    Input: Raw AirBnB listings DataFrame (same format as training data)
    Output: Cleaned DataFrame with exactly 25 features, ready for model
    
    Important: This function NEVER drops rows.
    """
    original_row_count = len(df)
    
    # Step 1: Convert data types
    df = convert_data_types(df)
    
    # Step 2: Create engineered features
    df = create_features(df)
    
    # Step 3: Handle missing values (fill, never drop)
    df = handle_missing_values(df)
    
    # Step 4: Select the 25 features the model expects
    X = select_features(df)
    
    # Step 5: Final cleanup
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    X = X.astype(float)
    
    # Clip extreme values (same as training does)
    for col in X.columns:
        p99 = X[col].quantile(0.99)
        p1 = X[col].quantile(0.01)
        if p99 > p1:  # avoid issues with constant columns
            X[col] = X[col].clip(lower=p1, upper=p99)
    
    # Verify we didn't lose any rows
    final_row_count = len(X)
    assert final_row_count == original_row_count, \
        f"Row count changed! Started with {original_row_count}, ended with {final_row_count}"
    
    print(f"Preprocessed {final_row_count} rows with {len(FEATURE_COLUMNS)} features")
    
    return X


# For testing
if __name__ == "__main__":
    print("Testing preprocessing_inference.py")
    print(f"Expected features: {len(FEATURE_COLUMNS)}")
    
    # Create minimal test data
    test_data = pd.DataFrame({
        'accommodates': [2, 4, 3],
        'bathrooms': [1, None, 2],
        'bedrooms': [1, 2, None],
        'beds': [1, None, 2],
        'price': ['$100', '$200', '$150'],
        'host_is_superhost': ['t', 'f', None],
        'instant_bookable': ['t', 'f', 't'],
        'minimum_nights': [1, 2, 3],
        'maximum_nights': [365, 30, 100],
        'latitude': [34.05, 40.71, None],
        'longitude': [-118.25, -74.01, None],
        'availability_365': [200, 100, 50],
        'description': ['Nice place', None, 'Great apartment with view'],
        'host_response_time': ['within an hour', None, 'within a day'],
        'neighborhood_overview': ['Great area', None, 'Downtown'],
        'host_about': ['I love hosting', None, None],
        'host_total_listings_count': [1, 5, None],
        'host_listings_count': [1, 3, None],
    })
    
    print(f"\nInput rows: {len(test_data)}")
    
    result = preprocess_for_inference(test_data)
    
    print(f"Output rows: {len(result)}")
    print(f"Output columns: {len(result.columns)}")
    print(f"\nColumns match expected: {list(result.columns) == FEATURE_COLUMNS}")
    print("\nFirst row sample:")
    print(result.iloc[0])
    print("\nTest passed!")