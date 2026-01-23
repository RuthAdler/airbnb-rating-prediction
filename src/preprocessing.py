"""
Data preprocessing and cleaning module for AirBnB rating prediction project.

Handles:
- Data type conversions
- Duplicate entries
- Missing values
- Outliers
- Feature engineering
"""

import ast
import pandas as pd
import numpy as np

# TODO: Add cleaning functions from notebook

def convert_data_types(df):
    # datetime columns
    date_cols = ['last_scraped', 'host_since', 'first_review', 'last_review']
    df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='coerce')

    # numeric columns
    numeric_cols = ['host_response_rate', 'host_acceptance_rate', 'price']
    df[numeric_cols] = (df[numeric_cols].replace(r'\s*[$%,]\s*', '', regex=True).apply(pd.to_numeric, errors='coerce'))

    # boolean columns
    bool_cols = ['host_is_superhost', 'host_has_profile_pic', 'instant_bookable']
    df[bool_cols] = df[bool_cols].replace({'t': True, 'f': False})

    # list columns
    df['host_verifications'] = df['host_verifications'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df['amenities'] = df['amenities'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    return df


def remove_duplicates(df):
    df = df.drop_duplicates()
    return df


def handle_missing_values(df, target_col='review_scores_rating') -> pd.DataFrame:
    """Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target variable column
        
    Returns:
        DataFrame with missing values handled
    """
    initial_rows = len(df)
    
    # Drop rows with missing target variable
    df = df.dropna(subset=[target_col])
    after_target_drop = len(df)

    # Remove rows missing all critical host info columns
    host_critical_cols = ['host_name', 'host_since', 'host_verifications', 
                          'host_has_profile_pic', 'host_total_listings_count', 
                          'host_listings_count']
    df = df.dropna(subset=host_critical_cols, how='any')

    # Fill missing values for minimum/maximum nights
    df['minimum_minimum_nights'] = df['minimum_minimum_nights'].fillna(df['minimum_nights'])
    df['maximum_minimum_nights'] = df['maximum_minimum_nights'].fillna(df['minimum_nights'])
    df['minimum_maximum_nights'] = df['minimum_maximum_nights'].fillna(df['maximum_nights'])
    df['maximum_maximum_nights'] = df['maximum_maximum_nights'].fillna(df['maximum_nights'])
    
    # Recover bathrooms data from bathrooms_text using mode
    if 'bathrooms' in df.columns and 'bathrooms_text' in df.columns:
        df['bathrooms'] = df.groupby('bathrooms_text')['bathrooms'].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))
        # Remove remaining rows with missing bathrooms
        df = df.loc[df['bathrooms'].notna()]

    # Fill missing bedrooms and beds with general and groups median values, respectively
    df['bedrooms'] = df.groupby('accommodates')['bedrooms'].transform(lambda x: x.fillna(x.median()))
    df['beds'] = df.groupby('accommodates')['beds'].transform(lambda x: x.fillna(x.median()))

    # Fill missing host_is_superhost with False (not superhost if missing) 
    df['host_is_superhost'] = df['host_is_superhost'].fillna(False)

    # Encode host_response_time
    mapping_dict = {'within an hour': 1, 'within a few hours': 2, 'within a day': 3, 'a few days or more': 4}
    df['host_response_time_coded'] = df['host_response_time'].map(mapping_dict).fillna(0)

    # Impute host response rate and acceptance rate with median values
    for col in ['host_response_rate', 'host_acceptance_rate']:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # Create binary indicators for presence of textual features
    df['has_neighborhood_overview'] = df['neighborhood_overview'].notna()
    df['has_host_about'] = df['host_about'].notna()

    return df

# TODO: Add outliers handling functions
def handle_outliers(df):
    '''Handle outliers in the dataset by applying log transformations.'''
    df['log_price'] = np.log(df['price'])
    df['log_price'] = df.groupby(['accommodates', 'city'])['log_price'].transform(lambda x: x.fillna(x.mean()))
    df['log_host_total_listings_count'] = np.log(df['host_total_listings_count'])
    df['log_host_listings_count'] = np.log(df['host_listings_count'])
    return df

# TODO: Add feature engineering functions

# TODO: Remove redundant columns function