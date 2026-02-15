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
}

CORE_COLS = {
    "accommodates",
    "bathrooms",
    "host_days_log",
    "is_superhost",
}

TOP_FEATURES = {
    "is_superhost",
    "host_days_log",
    "room_ratio",
    "minimum_nights",
}


DATE_COLS = {"last_scraped", "host_since", "first_review", "last_review"}



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
