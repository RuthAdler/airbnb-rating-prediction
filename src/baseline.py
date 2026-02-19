from sklearn.dummy import DummyRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_training_data(data_dir: str) -> pd.DataFrame:
    data_path = PROJECT_ROOT / data_dir
    datasets = []

    for csv_file in data_path.glob("listings*.csv"):
        if "TEST" in csv_file.name.upper():
            continue

        print(f"Loading {csv_file.name}...")
        df = pd.read_csv(csv_file)
        datasets.append(df)

    if not datasets:
        raise FileNotFoundError(
            f"No listings CSV files found in {data_path.resolve()}"
        )

    combined = pd.concat(datasets, ignore_index=True)
    print(f"Total rows: {len(combined)}")

    return combined

def train_dummy_model(X, y):
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    return model


def main(data_dir: str):
    """Train and save the model."""

    # Load data
    df = load_training_data(data_dir)
    df = df.dropna(subset=["review_scores_rating"])

    # Prepare features
    X = df
    y = df["review_scores_rating"].values

    model = train_dummy_model(X, y)

    #Evaluate on training set
    y_train_pred = model.predict(X)
    rmse_train = np.sqrt(mean_squared_error(y, y_train_pred))
    print(f"Train RMSE: {rmse_train:.4f}")

    #model mean
    mean = np.mean(y)
    print(f"Mean: {mean:.4f}")

    # Evaluate on test set
    X_test =  PROJECT_ROOT / "data/test/TEST_SET_X.csv"
    y_test = PROJECT_ROOT / "data/test/TEST_SET_y.csv"
    X_test = pd.read_csv(X_test)
    y_test = pd.read_csv(y_test)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main("data")
