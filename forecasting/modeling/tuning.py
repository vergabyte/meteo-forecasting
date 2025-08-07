import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from forecasting.modeling.evaluation import evaluate_forecast
from forecasting.modeling.predict import holt_winters_forecast

space = [
    Real(0.2, 0.8, name="alpha"),
    Real(0.0, 0.2, name="beta"),
    Real(0.01, 0.3, name="gamma"),
]


def create_objective(y_train: pd.Series, y_test: pd.Series, seasonal_periods: int):
    @use_named_args(space)
    def objective(alpha: float, beta: float, gamma: float) -> float:
        try:
            y_pred = holt_winters_forecast(
                y_train,
                forecast_horizon=len(y_test),
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                seasonal_periods=seasonal_periods,
            )
            if not np.isfinite(y_pred).all():
                return np.inf
            return evaluate_forecast(y_test, y_pred)["rmse"]
        except Exception:
            return np.inf

    return objective


def bayesian_tune(
    y_train: pd.Series,
    y_test: pd.Series,
    seasonal_periods: int = 365,
    n_calls: int = 30,
    random_state: int = 42,
) -> tuple[dict, pd.Series, object]:
    objective = create_objective(y_train, y_test, seasonal_periods)
    result = gp_minimize(
        objective, space, n_calls=n_calls, random_state=random_state, verbose=False
    )
    params = dict(zip(["alpha", "beta", "gamma"], result.x))
    forecast = holt_winters_forecast(
        y_train, forecast_horizon=len(y_test), **params, seasonal_periods=seasonal_periods
    )
    return params, forecast, result
