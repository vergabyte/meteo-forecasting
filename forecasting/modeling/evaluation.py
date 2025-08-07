import mlflow
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series, method_name: str = "unknown") -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }


def log_to_mlflow(
    method_name: str,
    metrics: dict,
    params: dict = None,
    experiment: str = "forecasting_comparison",
):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=method_name):
        mlflow.log_param("model", method_name)
        for k, v in (params or {}).items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
