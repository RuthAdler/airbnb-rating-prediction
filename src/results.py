from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred):
    """Evaluate the model's performance using MAE and RMSE metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred)
    }

