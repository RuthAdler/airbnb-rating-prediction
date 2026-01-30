import pandas as pd
from sklearn.dummy import DummyRegressor

from src.data_loading import load_all_listings
from src.preprocessing import preprocess_data
from src.train import train
from src.predict import predict
from src.results import evaluate

import pickle

MODEL_PATH="models/dummy_model.pkl"

def main():
    #Load data
    datasets = load_all_listings("data")
    df = pd.concat(datasets.values(), ignore_index=True)


    X_train, X_test, y_train, y_test= preprocess_data(df)

    # Train
    model = DummyRegressor(strategy="mean")
    model = train(model,X_train, y_train)

    # Save the model to a pickle file
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)


    #Predict
    y_pred = predict(model, X_test)

    #Evaluate
    metrics = evaluate(y_test, y_pred)
    print(metrics)

if __name__ == "__main__":
    main()
