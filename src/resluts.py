from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(model, X, y):
    """Evaluate the model's performance using MAE and RMSE metrics."""
    preds = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, preds),
        "RMSE": mean_squared_error(y, preds)
    }
