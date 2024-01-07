import plotly.express as px
import polars as pl
import streamlit as st

from amazon_product_search.core import source
from demo.page_config import set_page_config
from demo.utils import analyze_dataframe


def main() -> None:
    set_page_config()
    st.write("## Relevance Judgements")

    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    df = source.load_merged(locale)

    st.write("### Columns")
    analyzed_df = analyze_dataframe(df)
    st.write(analyzed_df.to_pandas())

    count_df = df.groupby("esci_label").count().sort("count", reverse=True)
    fig = px.bar(count_df.to_pandas(), x="esci_label", y="count")
    fig.update_layout(title="The number of labels")
    st.plotly_chart(fig)

    queries = sorted(df.get_column("query").unique().to_list())
    query = st.selectbox("Query:", queries)
    filtered_df = df.filter(pl.col("query") == query)

    st.write("### Examples")
    st.write(filtered_df.to_pandas())


if __name__ == "__main__":
    main()
