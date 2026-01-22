
def predict(model, X_test):
    """Generate predictions using the trained model."""
    predictions = model.predict(X_test)
    return predictions