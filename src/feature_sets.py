"""
Feature set selection for different dataset versions.
Works with BOTH:
- engineered features (features.py)
- cleaned raw features
"""

import pandas as pd

# ---------------- FEATURE GROUPS ----------------

TEXT_COLS = {
    "desc_length",
    "has_description",
    "mentions_clean",
    "mentions_luxury",
    "mentions_view",
    "mentions_location",
    "mentions_modern",
    "name_length",
}

GEO_COLS = {
    "latitude",
    "longitude",
}

HOST_COLS = {
    "host_days_log",
    "host_response_rate",
    "host_acceptance_rate",
    "is_superhost",
    "response_speed",
    "has_host_about",
    "has_neighborhood",
}

PROPERTY_COLS = {
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "room_ratio",
    "minimum_nights",
    "instant_bookable",
    "room_type",
    "property_type",
}

CORE_COLS = {
    "accommodates",
    "bathrooms",
    "host_days_log",
    "is_superhost",
}

PRICE_COLS = {'log_price'}

TOP_FEATURES = {
    "is_superhost",
    "host_days_log",
    "room_ratio",
    "minimum_nights",
}

DATE_COLS = {"last_scraped", "host_since", "first_review", "last_review"}

TEXT_ONLY = TEXT_COLS

GEO_ONLY = GEO_COLS

HOST_BEHAVIOR = {
    "host_response_rate",
    "host_acceptance_rate",
    "response_speed",
}

HOST_EXPERIENCE = {
    "host_days_log",
    "is_superhost",
}

BOOKING_FRICTION = {
    "minimum_nights",
    "instant_bookable",
}

CAPACITY = {
    "accommodates",
    "bedrooms",
    "beds",
    "room_ratio",
}

QUALITY_SIGNAL = {
    "is_superhost",
    "host_days_log",
    "mentions_clean",
    "desc_length",
}

HOST_NO_SUPERHOST = HOST_COLS - {"is_superhost"}



# ---------------- MAIN FUNCTION ----------------

def apply_feature_set(X: pd.DataFrame, dataset_version: str) -> pd.DataFrame:
    v = dataset_version.lower().strip()

    cols = list(X.columns)
    cols_set = set(cols)

    # remove unwanted columns safely
    cols_set -= DATE_COLS

    # ---------- VERSION LOGIC ----------

    if v in {"v0", "all"}:
        keep = cols_set

    elif v in {"v1", "no_text"}:
        keep = cols_set - TEXT_COLS

    elif v in {"v2", "no_geo"}:
        keep = cols_set - GEO_COLS

    elif v in {"v3", "host_only"}:
        keep = HOST_COLS

    elif v in {"v4", "property_only"}:
        keep = PROPERTY_COLS

    elif v in {"v5", "core"}:
        keep = CORE_COLS

    elif v in {"v6", "no_host"}:
        keep = cols_set - HOST_COLS

    elif v in {"v7", "human"}:
        keep = HOST_COLS | TEXT_COLS

    elif v in {"v8", "structure"}:
        keep = PROPERTY_COLS | GEO_COLS

    elif v in {"v9", "top"}:
        keep = TOP_FEATURES

    elif v in {"v10", "text_only"}:
        keep = TEXT_ONLY

    elif v in {"v11", "geo_only"}:
        keep = GEO_ONLY

    elif v in {"v12", "host_behavior"}:
        keep = HOST_BEHAVIOR

    elif v in {"v13", "host_experience"}:
        keep = HOST_EXPERIENCE

    elif v in {"v14", "booking"}:
        keep = BOOKING_FRICTION

    elif v in {"v15", "capacity"}:
        keep = CAPACITY

    elif v in {"v16", "quality_signal"}:
        keep = QUALITY_SIGNAL

    elif v in {"v17", "no_structure"}:
        keep = cols_set - PROPERTY_COLS

    elif v in {"v18", "host_property"}:
        keep = HOST_COLS | PROPERTY_COLS

    elif v in {"v19", "host_no_superhost"}:
        keep = HOST_NO_SUPERHOST

    elif v in {"v20", "host_text"}:
        keep = HOST_COLS | TEXT_COLS

    elif v in {"v25", "no_geo_no_price"}:
        keep = cols_set - GEO_COLS - PRICE_COLS



    else:
        raise ValueError(f"Unknown dataset_version '{dataset_version}'")

    # keep only columns that actually exist
    keep_existing = [c for c in cols if c in keep and c in cols_set]

    if len(keep_existing) == 0:
        raise ValueError(
            f"No features selected for version '{dataset_version}'. "
            "Check feature pool."
        )

    print(f"[FeatureSet {v}] using {len(keep_existing)} features")

    return X[keep_existing]
