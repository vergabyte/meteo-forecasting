# Meteo Forecasting

Time series forecasting for weather data using statistical methods such as Naive Forecasting, Moving Average, and Holt-Winters Exponential Smoothing.

## Requirements

- Python 3.11+
- Conda

## Setup

```bash
git clone https://github.com/vergabyte/meteo-forecasting.git
cd meteo_forecasting
conda env create -f environment.yml
conda activate forecasting
pip install -e .
```

## Project Structure

```
├── data/              # Raw, processed, interim, and external data
├── forecasting/       # Forecasting logic (config, utils, modeling, plots)
├── models/            # Serialized models
├── notebooks/         # Exploratory and development notebooks
├── reports/           # Generated reports and figures
├── environment.yml    # Conda environment
├── pyproject.toml     # Project metadata
```

## Optional

To launch MLflow UI:

```bash
mlflow ui
# Then open http://localhost:5000
```
