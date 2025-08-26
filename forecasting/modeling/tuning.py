import numpy as np
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


def create_objective(y_train, y_test, seasonal_periods):
    @use_named_args(space)
    def objective(alpha, beta, gamma):
        try:
            y_pred = holt_winters_forecast(
                y_train, len(y_test), alpha, beta, gamma, seasonal_periods
            )
            if not np.isfinite(y_pred).all():
                return np.inf
            return evaluate_forecast(y_test, y_pred)["rmse"]
        except Exception:
            return np.inf

    return objective


def bayesian_tune(y_train, y_test, seasonal_periods, n_calls, random_state):
    objective = create_objective(y_train, y_test, seasonal_periods)
    result = gp_minimize(
        objective, space, n_calls=n_calls, random_state=random_state, verbose=False
    )
    params = dict(zip(["alpha", "beta", "gamma"], result.x))
    forecast = holt_winters_forecast(
        y_train, len(y_test), params["alpha"], params["beta"], params["gamma"], seasonal_periods
    )
    return params, forecast, result
