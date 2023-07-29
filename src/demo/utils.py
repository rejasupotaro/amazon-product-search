import polars as pl
import streamlit as st

from amazon_product_search.core import source
from amazon_product_search.core.source import Locale


@st.cache_data
def load_products(locale: Locale, nrows: int = -1) -> pl.DataFrame:
    return source.load_products(locale, nrows)


@st.cache_data
def load_labels(locale: Locale, nrows: int = -1) -> pl.DataFrame:
    return source.load_labels(locale, nrows)


@st.cache_data
def load_merged(locale: Locale, nrows: int = -1) -> pl.DataFrame:
    return source.load_merged(locale, nrows)


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
