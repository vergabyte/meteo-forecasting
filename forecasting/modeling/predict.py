import numpy as np
import pandas as pd


def naive_forecast(y: pd.Series, forecast_horizon: int) -> pd.Series:
    value = y.iloc[-1]
    index = pd.date_range(start=y.index[-1] + pd.Timedelta(1, "D"), periods=forecast_horizon)
    return pd.Series([value] * forecast_horizon, index=index)


def moving_average_forecast(y: pd.Series, forecast_horizon: int, window_size: int) -> pd.Series:
    if len(y) < window_size:
        raise ValueError(f"Need at least {window_size} values to compute average")

    average = y.iloc[-window_size:].mean()
    index = pd.date_range(start=y.index[-1] + pd.Timedelta(1, "D"), periods=forecast_horizon)
    return pd.Series([average] * forecast_horizon, index=index)


def ses_forecast(y: pd.Series, forecast_horizon: int = 1, alpha: float = 0.5) -> pd.Series:
    if not 0 < alpha < 1 or y.empty:
        raise ValueError("alpha must be in (0, 1) and y must be non-empty")

    s = y.iloc[0]
    for val in y[1:]:
        s = alpha * val + (1 - alpha) * s

    index = pd.date_range(y.index[-1] + pd.Timedelta(1, "D"), periods=forecast_horizon)
    return pd.Series([s] * forecast_horizon, index=index)


def holt_trend_forecast(
    y: pd.Series, forecast_horizon: int, alpha: float, beta: float
) -> pd.Series:
    if len(y) < 2:
        raise ValueError("Need at least two data points to initialize.")

    level, trend = y.iloc[0], y.iloc[1] - y.iloc[0]

    for t in range(1, len(y)):
        prev_level = level
        level = alpha * y.iloc[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    forecast = [level + h * trend for h in range(1, forecast_horizon + 1)]
    index = pd.date_range(y.index[-1] + pd.Timedelta(1, "D"), periods=forecast_horizon)

    return pd.Series(forecast, index=index)


def holt_winters_forecast(
    y: pd.Series,
    forecast_horizon: int,
    alpha: float,
    beta: float,
    gamma: float,
    seasonal_periods: int,
) -> pd.Series:
    if len(y) < 2 * seasonal_periods:
        raise ValueError("Need at least two full seasonal cycles.")

    y, n = y.copy(), len(y)
    level = np.zeros(n)
    trend = np.zeros(n)
    seasonals = np.array(
        [y.iloc[i] - y.iloc[:seasonal_periods].mean() for i in range(seasonal_periods)]
    )

    sp = seasonal_periods - 1
    level[sp] = y.iloc[:seasonal_periods].mean()
    trend[sp] = np.mean(
        [
            (y.iloc[i + seasonal_periods] - y.iloc[i]) / seasonal_periods
            for i in range(seasonal_periods)
        ]
    )

    for t in range(seasonal_periods, n):
        s = t % seasonal_periods
        level[t] = alpha * (y.iloc[t] - seasonals[s]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        seasonals[s] = gamma * (y.iloc[t] - level[t]) + (1 - gamma) * seasonals[s]

    last_level, last_trend = level[-1], trend[-1]
    forecast = [
        last_level + h * last_trend + seasonals[(n + h - 1) % seasonal_periods]
        for h in range(1, forecast_horizon + 1)
    ]
    return pd.Series(
        forecast,
        index=pd.date_range(start=y.index[-1] + pd.Timedelta(1, "D"), periods=forecast_horizon),
    )
