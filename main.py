import pandas as pd
from sklearn.dummy import DummyRegressor

from src.data_loading import load_all_listings
from src.geo_processing import cluster_coordinates, distance_to_center
#TODO: from src.preprocessing import preprocess_data
from src.train import train
from src.predict import predict
from src.results import evaluate
from src.visualization import correlation_heatmap


def main():
    #Load data
    datasets = load_all_listings("data")
    df = pd.concat(datasets.values(), ignore_index=True)


    #TODO: Preprocessing
    X_train, X_test, y_train, y_test= None, None, None, None

    # Train
    model = DummyRegressor(strategy="mean")
    model = train(X_train, y_train)

    #Predict
    preds = predict(model, X_test)

    #Evaluate
    metrics = evaluate(model, X_test, y_test)
    print(metrics)

    #Visualize
    correlation_heatmap(df)


if __name__ == "__main__":
    main()
