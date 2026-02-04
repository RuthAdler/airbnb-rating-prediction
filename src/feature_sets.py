"""
Feature set selection for different dataset versions.
"""

import pandas as pd

TEXT_COLS = {"description_length_words", "description_length_chars"}

GEO_COLS = {"distance_to_center"}

HOST_COLS = {
    "host_tenure_days",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_response_time_coded",
    "has_neighborhood_overview",
    "has_host_about",
    "log_host_total_listings_count",
    "log_host_listings_count",
}

PROPERTY_COLS = {
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "minimum_minimum_nights",
    "estimated_occupancy_l365d",
    "instant_bookable",
    "log_price",
}

LISTING_EFFORT_COLS = {
    "description_length_words",
    "description_length_chars",
    "has_host_about",
    "has_neighborhood_overview",
}


STRUCTURAL_COLS = {
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
}

CORE_COLS = {
    "log_price",
    "accommodates",
    "bathrooms",
    "distance_to_center",
    "host_tenure_days",
    "host_is_superhost",
}

TOP_FEATURES = {
    "host_is_superhost",
    "log_host_total_listings_count",
    "host_tenure_days",
    "minimum_maximum_nights",
}


DATE_COLS = {"last_scraped", "host_since", "first_review", "last_review"}
REDUNDANT= {"latitude", "longitude"}


def apply_feature_set(X: pd.DataFrame, dataset_version: str) -> pd.DataFrame:
    v = dataset_version.lower().strip()
    cols = list(X.columns)
    cols_set = set(cols)

    # Safety: remove raw date columns if they survived
    cols_set -= (DATE_COLS & cols_set | REDUNDANT & cols_set)

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

    elif v in {"v8", "price_only"}:
        keep = {"log_price", "estimated_occupancy_l365d"}

    elif v in {"v9"}:
        keep = TOP_FEATURES

    else:
        raise ValueError(
            f"Unknown dataset_version='{dataset_version}'.\n"
            "Valid options:\n"
            "v0_all_features\n"
            "v1_no_text\n"
            "v2_no_geo\n"
            "v3_host_only\n"
            "v4_property_only\n"
            "v5_core\n"
            "v6_no_host\n"
            "v7_human_factors\n"
            "v8_price_only"
        )

    keep_ordered = [c for c in cols if c in keep]
    return X[keep_ordered]