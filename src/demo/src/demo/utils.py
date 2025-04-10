import streamlit as st
from data_source import Locale, loader
from pandas import DataFrame
from pandas.api.types import is_object_dtype


@st.cache_data
def load_products(locale: Locale, nrows: int = -1) -> DataFrame:
    return loader.load_products("../data-source/data", locale)[:nrows]


@st.cache_data
def load_labels(locale: Locale, nrows: int = -1) -> DataFrame:
    return loader.load_examples("../data-source/data", locale)[:nrows]


def analyze_dataframe(df: DataFrame) -> DataFrame:
    """Calculate the basic statistics for a given DataFrame.

    Args:
        df (DataFrame): A DataFrame to analyze.

    Returns:
        pd.DataFrame: The resulting DataFrame containing column, dtype, nan_rate, and mean_length.
    """
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "nan_rate": series.isnull().sum() / len(series),
                "mean_length": series.fillna("").apply(len).mean() if is_object_dtype(series) else None,
            }
        )
    return DataFrame(rows)
