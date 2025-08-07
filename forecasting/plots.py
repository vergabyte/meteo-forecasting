import matplotlib.pyplot as plt


def plot_time_series(data, title="Time Series Plot", xlabel="Time", ylabel=None):
    ax = data.plot(figsize=(12, 4), title=title)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_forecasts(y_true, forecasts, title="Forecast Comparison"):
    plt.figure(figsize=(12, 4))
    y_true.plot(label="Actual", color="black", linewidth=1.5)

    for name, y_pred in forecasts.items():
        y_pred.plot(label=name, linestyle="--", linewidth=2)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_forecast_with_train(y_train, y_test, y_pred, title="Forecast vs Actual"):
    plt.figure(figsize=(12, 4))
    plt.plot(y_train.index, y_train, label="Train", linestyle="--")
    plt.plot(y_test.index, y_test, label="Actual", color="black")
    plt.plot(y_pred.index, y_pred, label="Forecast", color="tab:red")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
