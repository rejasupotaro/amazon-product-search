import plotly.express as px
import streamlit as st
from data_source import loader

from demo.page_config import set_page_config
from demo.utils import analyze_dataframe


def main() -> None:
    set_page_config()
    st.write("## Relevance Judgements")

    locale = st.selectbox("Locale:", ["jp", "us", "es"])
    df = loader.load_examples(locale=locale)

    st.write("### Columns")
    analyzed_df = analyze_dataframe(df)
    st.write(analyzed_df)

    count_df = df.groupby("esci_label").size().reset_index(name="count").sort_values("count", ascending=False)
    fig = px.bar(count_df, x="esci_label", y="count")
    fig.update_layout(title="The number of labels")
    st.plotly_chart(fig)

    queries = sorted(df["query"].unique().tolist())
    query = st.selectbox("Query:", queries)
    filtered_df = df[df["query"] == query]

    st.write("### Examples")
    st.write(filtered_df)


if __name__ == "__main__":
    main()
