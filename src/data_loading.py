"""
Data loading module for AirBnB rating prediction project.
"""

import pandas as pd
from pathlib import Path


def load_listings(filepath: str) -> pd.DataFrame:
    """Load a single AirBnB listings CSV file."""
    return pd.read_csv(filepath)


def load_all_listings(data_dir: str = "data") -> dict:
    """Load all listing CSV files from a directory."""
    data_path = Path(data_dir)
    datasets = {}
    
    for csv_file in data_path.glob("listings*.csv"):
        city_name = csv_file.stem.replace("listings", "").strip()
        datasets[city_name] = pd.read_csv(csv_file)
        print(f"Loaded {city_name}: {datasets[city_name].shape[0]} rows")
    
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