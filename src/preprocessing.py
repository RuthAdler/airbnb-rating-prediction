"""
Data preprocessing and cleaning module for AirBnB rating prediction project.

Handles:
- Missing values
- Outliers
- Data type conversions
- Feature cleaning
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

    # Fill missing bedrooms and beds with median values
    df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
    df['beds'] = df['beds'].fillna(df['beds'].median())

    # Fill missing host_is_superhost with False (not superhost if missing) 
    df['host_is_superhost'] = df['host_is_superhost'].fillna(False)

# TODO: Add outliers handling functions

# TODO: Add feature engineering functions
