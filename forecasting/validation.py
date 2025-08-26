import pandas as pd


def check_column_no_nan(df, column):
    if df[column].isna().any():
        raise ValueError(f"{column} contains NaN")


def check_column_numeric(df, column):
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"{column} is not numeric")


def check_index_type(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")


def check_index_no_missing(df):
    if df.index.hasnans:
        raise ValueError("Index has NaT values")


def check_index_no_duplicates(df):
    if df.index.duplicated().any():
        raise ValueError("Index has duplicates")


def check_index_regular_frequency(df):
    freq = pd.infer_freq(df.index)
    if freq is None:
        raise ValueError("Index has no regular frequency")
