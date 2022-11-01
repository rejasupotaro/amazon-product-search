import pandas as pd
from pandas.api.types import is_string_dtype


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
