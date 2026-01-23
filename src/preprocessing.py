"""
Data preprocessing and cleaning module for AirBnB rating prediction project.

Handles:
- Data type conversions
- Duplicate entries
- Missing values
- Outliers
- Feature engineering
"""

import os
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


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

def split_data(df, test_size=0.25, random_state=42):
    # Drop rows with missing target variable
    df = df.dropna(subset=['review_scores_rating'])
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_data, test_data

def handle_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit imputers on train data, apply to both train and test."""
    
    # 1. Independent operations (row dropping & constant filling)
    host_critical_cols = ['host_name', 'host_since', 'host_verifications', 
                          'host_has_profile_pic', 'host_total_listings_count', 
                          'host_listings_count']
    
    for df in [train_df, test_df]:
        df.dropna(subset=host_critical_cols, how='any', inplace=True)
        df['minimum_minimum_nights'] = df['minimum_minimum_nights'].fillna(df['minimum_nights'])
        df['maximum_minimum_nights'] = df['maximum_minimum_nights'].fillna(df['minimum_nights'])
        df['minimum_maximum_nights'] = df['minimum_maximum_nights'].fillna(df['maximum_nights'])
        df['maximum_maximum_nights'] = df['maximum_maximum_nights'].fillna(df['maximum_nights'])
        df['host_is_superhost'] = df['host_is_superhost'].fillna(False)
        
        mapping_dict = {'within an hour': 1, 'within a few hours': 2, 'within a day': 3, 'a few days or more': 4}
        df['host_response_time_coded'] = df['host_response_time'].map(mapping_dict).fillna(0)
        df['has_neighborhood_overview'] = df['neighborhood_overview'].notna()
        df['has_host_about'] = df['host_about'].notna()

    # 2. Leakage-sensitive operations (learn from train, apply to both)
    # Bathrooms
    if 'bathrooms' in train_df.columns and 'bathrooms_text' in train_df.columns:
        baths_map = train_df.groupby('bathrooms_text')['bathrooms'].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        train_df['bathrooms'] = train_df['bathrooms'].fillna(train_df['bathrooms_text'].map(baths_map))
        test_df['bathrooms'] = test_df['bathrooms'].fillna(test_df['bathrooms_text'].map(baths_map))
        train_df.dropna(subset=['bathrooms'], inplace=True)
        test_df.dropna(subset=['bathrooms'], inplace=True)

    # Bedrooms/Beds
    for col in ['bedrooms', 'beds']:
        median_map = train_df.groupby('accommodates')[col].median()
        train_df[col] = train_df[col].fillna(train_df['accommodates'].map(median_map))
        test_df[col] = test_df[col].fillna(test_df['accommodates'].map(median_map))

    # Response/Acceptance rates
    for col in ['host_response_rate', 'host_acceptance_rate']:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

    return train_df, test_df

def handle_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Handle outliers, learning group means from train_df only."""
    for df in [train_df, test_df]:
        df['log_price'] = np.log(df['price'])
        df['log_host_total_listings_count'] = np.log(df['host_total_listings_count'] + 1)
        df['log_host_listings_count'] = np.log(df['host_listings_count'] + 1)
    
    # Fill remaining NaN log_prices using train_df means
    price_mean_map = train_df.groupby(['accommodates', 'city'])['log_price'].mean()
    train_df['log_price'] = train_df['log_price'].fillna(train_df.set_index(['accommodates', 'city']).index.map(price_mean_map))
    test_df['log_price'] = test_df['log_price'].fillna(test_df.set_index(['accommodates', 'city']).index.map(price_mean_map))
    
    return train_df, test_df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['description_length_words'] = df['description'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    df['description_length_chars'] = df['description'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    return df

# TODO: Remove redundant and non-numeric columns
def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are no longer needed after processing."""
    cols_to_drop = [
        'price', 'host_total_listings_count', 'host_listings_count', # Replaced by log versions
        'description', 'neighborhood_overview', 'host_about', # Replaced by text features
        'bathrooms_text', 'host_response_time' # Replaced by coded versions
    ]
    non_numeric_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df = df.drop(columns=cols_to_drop + non_numeric_cols, errors='ignore')
    return df

def preprocess_data(df: pd.DataFrame, save_dir: str = 'data/processed') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main preprocessing function that saves intermediate CSVs at each step."""
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # 1. Pre-split cleaning
    df = convert_data_types(df)
    df = remove_duplicates(df)

    # 2. Split and Save Initial Split
    train_df, test_df = split_data(df)
    train_df.to_csv(f"{save_dir}/1_train_split.csv", index=False)
    test_df.to_csv(f"{save_dir}/1_test_split.csv", index=False)
    print("Saved 1_train_split.csv and 1_test_split.csv")

    # 3. Handle Missing Values and Save
    train_df, test_df = handle_missing_values(train_df, test_df)
    train_df.to_csv(f"{save_dir}/2_train_imputed.csv", index=False)
    test_df.to_csv(f"{save_dir}/2_test_imputed.csv", index=False)
    print("Saved 2_train_imputed.csv and 2_test_imputed.csv")

    # 4. Outliers & Feature Engineering and Save
    train_df, test_df = handle_outliers(train_df, test_df)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    train_df.to_csv(f"{save_dir}/3_train_engineered.csv", index=False)
    test_df.to_csv(f"{save_dir}/3_test_engineered.csv", index=False)
    print("Saved 3_train_engineered.csv and 3_test_engineered.csv")

    # 5. Final Cleanup and Save Final
    train_df = remove_redundant_columns(train_df)
    test_df = remove_redundant_columns(test_df)
    train_df.to_csv(f"{save_dir}/4_train_final.csv", index=False)
    test_df.to_csv(f"{save_dir}/4_test_final.csv", index=False)
    print("Saved 4_train_final.csv and 4_test_final.csv")
    
    return train_df, test_df