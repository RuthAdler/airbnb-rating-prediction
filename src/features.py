import pandas as pd
import numpy as np

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
    'room_type',
    'property_type',

    # Text keywords (7)
    'mentions_clean',
    'mentions_luxury',
    'mentions_view',
    'mentions_location',
    'mentions_modern',
    'has_neighborhood',
    'name_length',
]


def simplify_property_type(x):
    x = str(x).lower()

    if "apartment" in x or "rental" in x or "loft" in x:
        return 1  # apartment-like

    if "house" in x or "home" in x or "villa" in x:
        return 2  # house-like

    if "condo" in x:
        return 3

    if "hotel" in x or "serviced" in x:
        return 4

    return 0  # other / rare


def build_feature_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Create FULL feature pool (for experiments)."""

    X = pd.DataFrame()

    # ---- Property ----
    X['accommodates'] = df['accommodates'].fillna(2)

    df["bathrooms"] = df["bathrooms"].fillna(
        df["bathrooms_text"].str.extract(r"(\d+\.?\d*)").astype(float)[0]
    )
    X['bathrooms'] = df['bathrooms'].fillna(1)

    X['bedrooms'] = df['bedrooms'].fillna(1)
    X['beds'] = df['beds'].fillna(1)
    X['room_ratio'] = X['bedrooms'] / X['accommodates'].clip(lower=1)

    # ---- GEO ----
    X["longitude"] = df["longitude"].fillna(0)
    X["latitude"] = df["latitude"].fillna(0)

    # ---- Host rates ----
    for col in ['host_response_rate', 'host_acceptance_rate']:
        X[col] = df[col].astype(str).str.replace('%', '').str.replace('N/A', '')
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(100) / 100

    X['is_superhost'] = df['host_is_superhost'].map({
        't': 1, 'f': 0, True: 1, False: 0
    }).fillna(0)

    df_copy = df.copy()
    df_copy['host_since'] = pd.to_datetime(df_copy['host_since'], errors='coerce')
    df_copy['last_scraped'] = pd.to_datetime(df_copy['last_scraped'], errors='coerce')

    host_days = (df_copy['last_scraped'] - df_copy['host_since']).dt.days.fillna(0).clip(lower=0)
    X['host_days_log'] = np.log1p(host_days)

    resp_map = {
        'within an hour': 1.0,
        'within a few hours': 0.75,
        'within a day': 0.5,
        'a few days or more': 0.25
    }
    X['response_speed'] = df['host_response_time'].map(resp_map).fillna(0.5)

    # ---- Booking ----
    X['minimum_nights'] = np.log1p(df['minimum_nights'].fillna(1).clip(0, 365))
    X['instant_bookable'] = df['instant_bookable'].map({
        't': 1, 'f': 0, True: 1, False: 0
    }).fillna(0)

    # ---- Text ----
    X['has_description'] = df['description'].notna().astype(int)
    X['desc_length'] = df['description'].fillna('').str.len().clip(0, 2000) / 2000
    X['has_host_about'] = df['host_about'].notna().astype(int)

    desc = df['description'].fillna('').str.lower()

    X['mentions_clean'] = desc.str.contains('clean|spotless|sanitize|hygien', regex=True).astype(int)
    X['mentions_luxury'] = desc.str.contains('luxury|premium|elegant', regex=True).astype(int)
    X['mentions_view'] = desc.str.contains('view|ocean|beach|lake', regex=True).astype(int)
    X['mentions_location'] = desc.str.contains('downtown|central|metro', regex=True).astype(int)
    X['mentions_modern'] = desc.str.contains('modern|renovated|updated', regex=True).astype(int)

    X['has_neighborhood'] = df['neighborhood_overview'].notna().astype(int)
    X['name_length'] = df['name'].fillna('').str.len().clip(0, 100) / 100

    # room and property types
    ROOM_TYPE_MAP = {
        "Entire home/apt": 2,
        "Private room": 1,
        "Shared room": 0,
        "Hotel room": 2
    }

    X["room_type"] = df["room_type"].map(ROOM_TYPE_MAP).fillna(1)

    X["property_type"] = df["property_type"].apply(simplify_property_type)

    return X.fillna(0)


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Production features used by final model only."""
    X = build_feature_pool(df)
    return X[FEATURE_COLUMNS]
