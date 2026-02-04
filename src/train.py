"""
Module for training machine learning models and saving them to disk.
"""
import joblib


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def train_and_save(model, X_train, y_train, path):
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    return model
