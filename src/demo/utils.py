import pandas as pd
import streamlit as st
from pandas.api.types import is_string_dtype

from amazon_product_search import source
from amazon_product_search.source import Locale


@st.cache
def load_products(locale: Locale, nrows: int = -1) -> pd.DataFrame:
    return source.load_products(locale, nrows)


@st.cache
def load_labels(locale: Locale, nrows: int = -1) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


@st.cache
def load_merged(locale: Locale, nrows: int = -1) -> pd.DataFrame:
    return source.load_merged(locale, nrows)


def split_fields(fields: list[str]) -> tuple[list[str], list[str]]:
    """Convert a given list of fields into a tuple of (sparse_fields, dense_fields)

    Field names containing "vector" will be considered dense_fields.

    Args:
        fields (list[str]): A list of fields.

    Returns:
        tuple[list[str], list[str]]: A tuple of (sparse_fields, dense_fields)
    """
    sparse_fields: list[str] = []
    dense_fields: list[str] = []
    for field in fields:
        (dense_fields if "vector" in field else sparse_fields).append(field)
    return sparse_fields, dense_fields


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the basic statistics for a given DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame to analyze.

    Returns:
        pd.DataFrame: The resulting DataFrame containing column, dtype, nan_rate, and mean_length.
    """
    columns = df.columns.to_list()
    rows = []
    for c in columns:
        series = df[c]

        length = None
        if is_string_dtype(series):
            length = series.fillna("").apply(len).mean()

        rows.append(
            {
                "column": c,
                "dtype": series.dtype.name,
                "nan_rate": series.isna().sum() / len(series),
                "mean_length": length,
            }
        )
    return pd.DataFrame(rows)
