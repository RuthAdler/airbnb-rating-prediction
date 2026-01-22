
def train(model,X_train,y_train):
    """Train the given model with the provided training data."""
    model.fit(X_train, y_train)
    return model


