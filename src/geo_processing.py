"""
Geographic processing module for AirBnB rating prediction project.
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from typing import Tuple

def cluster_coordinates(data: "pd.DataFrame") -> "pd.DataFrame":
    """Cluster listings based on their geographic coordinates."""

    X = data[["longitude", "latitude"]]
    kmeans = KMeans(n_clusters=5, random_state=42)
    data["area"] = kmeans.fit_predict(X)
    return data



def fit_city_center(train_df: pd.DataFrame) -> Tuple[float, float]:
    return train_df["longitude"].mean(), train_df["latitude"].mean()

def add_distance_to_center(df: pd.DataFrame, center: Tuple[float, float]) -> pd.DataFrame:
    lon0, lat0 = center
    df = df.copy()
    df["distance_to_center"] = np.sqrt((df["longitude"] - lon0)**2 + (df["latitude"] - lat0)**2)
    return df
