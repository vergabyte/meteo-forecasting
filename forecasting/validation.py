from loguru import logger
import pandas as pd
from pandas import DataFrame


def check_column_no_nan(df: DataFrame, column: str) -> None:
    if df[column].isna().any():
        raise ValueError(f"{column} contains NaN")
    logger.info(f"{column} has no NaN")


def check_column_numeric(df: DataFrame, column: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise TypeError(f"{column} is not numeric")
    logger.info(f"{column} is numeric")


def check_index_type(df: DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be DatetimeIndex")
    logger.info("Index is DatetimeIndex")


def check_index_no_missing(df: DataFrame) -> None:
    if df.index.hasnans:
        raise ValueError("Index has NaT values")
    logger.info("Index has no missing values")


def check_index_no_duplicates(df: DataFrame) -> None:
    if df.index.duplicated().any():
        raise ValueError("Index has duplicates")
    logger.info("Index has no duplicates")


def check_index_regular_frequency(df: DataFrame) -> None:
    freq = pd.infer_freq(df.index)
    if freq is None:
        raise ValueError("Index has no regular frequency")
    logger.info(f"Index frequency: {freq}")
