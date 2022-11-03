import pandas as pd
import streamlit as st
from pandas.api.types import is_string_dtype

from amazon_product_search import source


@st.cache
def load_products(locale: str, nrows: int = -1) -> pd.DataFrame:
    return source.load_products(locale, nrows)


@st.cache
def load_labels(locale: str, nrows: int = -1) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def analyze_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
