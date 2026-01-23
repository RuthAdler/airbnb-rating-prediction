"""
Data loading module for AirBnB rating prediction project.
"""

import pandas as pd
from pathlib import Path


def load_listings(filepath: str, city: str = None) -> pd.DataFrame:
    """Load a single AirBnB listings CSV file."""
    df = pd.read_csv(filepath)
    if city:
        df["city"] = city
    return df


def load_all_listings(data_dir: str = "data") -> dict:
    """Load all listing CSV files from a directory."""
    data_path = Path(data_dir)
    datasets = {}

    for csv_file in data_path.glob("listings*.csv"):
        city_name = csv_file.stem.replace("listings", "").strip()

        df = pd.read_csv(csv_file)
        df["city"] = city_name

        datasets[city_name] = df
        print(f"Loaded {city_name}: {df.shape[0]} rows")

    return datasets


def validate_columns_match(datasets: dict) -> bool:
    """Check if all datasets have the same columns."""
    if len(datasets) < 2:
        return True
    
    dataframes = list(datasets.values())
    first_cols = set(dataframes[0].columns)
    
    for df in dataframes[1:]:
        if set(df.columns) != first_cols:
            return False
    
    return True