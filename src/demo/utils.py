import polars as pl
import streamlit as st

from amazon_product_search import source
from amazon_product_search.source import Locale


@st.cache_data
def load_products(locale: Locale, nrows: int = -1) -> pl.DataFrame:
    return source.load_products(locale, nrows)


@st.cache_data
def load_labels(locale: Locale, nrows: int = -1) -> pl.DataFrame:
    return source.load_labels(locale, nrows)


@st.cache_data
def load_merged(locale: Locale, nrows: int = -1) -> pl.DataFrame:
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


def analyze_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the basic statistics for a given DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame to analyze.

    Returns:
        pd.DataFrame: The resulting DataFrame containing column, dtype, nan_rate, and mean_length.
    """
    rows = []
    for column in df.columns:
        series = df.get_column(column)
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "nan_rate": series.is_null().sum() / len(series),
                "mean_length": series.fill_null("").apply(len).mean() if series.dtype == pl.Utf8 else None,
            }
        )
    return pl.from_dicts(rows)
