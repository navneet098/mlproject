"""Evaluation helpers for WattWise forecasting outputs."""
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred):
    return {
        "MAE_kW": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_kW": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }
