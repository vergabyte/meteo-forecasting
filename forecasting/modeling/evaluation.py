import mlflow
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def evaluate_forecast(y_true, y_pred, method_name):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
    }


def log_to_mlflow(method_name, metrics, params, experiment):
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=method_name):
        mlflow.log_param("model", method_name)
        for k, v in (params or {}).items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
