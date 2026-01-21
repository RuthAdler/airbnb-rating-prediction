"""
Geographic processing module for AirBnB rating prediction project.

TODO: Current implementation uses NYC/LA-specific shapefiles.
      Need to make this generalizable for unknown cities.
"""

# TODO: Implement generalizable geo processing

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def cluster_coordinates(data: "pd.DataFrame") -> "pd.DataFrame":
    """Cluster listings based on their geographic coordinates."""

    X = data[["longitude", "latitude"]]
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    data["area"] = kmeans.fit_predict(X)
    return data

def distance_to_center(data: "pd.DataFrame") -> "pd.DataFrame":
    """Calculate distance from each listing to the city center."""

    city_center = (np.mean(data["longitude"]),np.min(data["latitude"]))
    data["distance_to_center"] = np.sqrt((data["longitude"] - city_center[0])**2 + (data["latitude"] - city_center[1])**2)
    return data