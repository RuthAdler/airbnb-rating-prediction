"""
Geographic processing module for AirBnB rating prediction project.

TODO: Current implementation uses NYC/LA-specific shapefiles.
      Need to make this generalizable for unknown cities.
"""

# TODO: Implement generalizable geo processing

from sklearn.cluster import KMeans
import pandas as pd


def cluster_coordinates(data: "pd.DataFrame") -> "pd.DataFrame":
    """Cluster listings based on their geographic coordinates."""

    X = data[["longitude", "latitude"]]
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    data["area"] = kmeans.fit_predict(X)
    return data

