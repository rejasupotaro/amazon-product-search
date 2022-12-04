import pandas as pd
import streamlit as st

from amazon_product_search.synonyms.synonym_dict import SynonymDict
from demo.page_config import set_page_config
from demo.utils import load_labels

SBERT_SYNONYM_DICT = SynonymDict(synonym_filename="synonyms_jp_sbert.csv")
FINE_TUNED_SBERT_SYNONYM_DICT = SynonymDict(synonym_filename="synonyms_jp_fine_tuned_sbert.csv")


def load_queries() -> list[str]:
    labels_df = load_labels(locale="jp")
    queries = labels_df.sample(frac=1)["query"].unique().tolist()
    return queries


def main():
    set_page_config()
    st.write("## Synonyms")

    queries = load_queries()[:1000]

    rows = []
    for query in queries:
        rows.append(
            {
                "query": query,
                "variant": "sbert",
                "synonyms": SBERT_SYNONYM_DICT.find_synonyms(query),
            }
        )
        rows.append(
            {
                "query": query,
                "variant": "fine_tuned_sbert",
                "synonyms": FINE_TUNED_SBERT_SYNONYM_DICT.find_synonyms(query),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df.head(100), use_container_width=True)

    df["num_synonyms"] = df["synonyms"].apply(len)

    stats_df = (
        df.groupby("variant")
        .agg(
            num_synonyms=("num_synonyms", lambda series: series.mean().round(4)),
        )
        .reset_index()
    )
    st.write(stats_df)


if __name__ == "__main__":
    main()
