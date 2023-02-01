import polars as pl
import streamlit as st

from amazon_product_search.synonyms.synonym_dict import SynonymDict
from demo.page_config import set_page_config
from demo.utils import load_labels

SBERT_SYNONYM_DICT = SynonymDict(synonym_filename="synonyms_jp_sbert.csv")
FINE_TUNED_SBERT_SYNONYM_DICT = SynonymDict(synonym_filename="synonyms_jp_fine_tuned_sbert.csv")


def load_queries() -> list[str]:
    labels_df = load_labels(locale="jp")
    queries = labels_df.get_column("query").sample(frac=1).unique().to_list()
    return queries


def main():
    set_page_config()
    st.write("## Synonyms")

    queries = load_queries()[:1000]

    rows = []
    for query in queries:
        for synonym_source, synonym_dict in [
            ("sbert", SBERT_SYNONYM_DICT),
            ("fine_tuned_sbert", FINE_TUNED_SBERT_SYNONYM_DICT),
        ]:
            for threshold in [0.6, 0.7, 0.8, 0.9]:
                rows.append(
                    {
                        "query": query,
                        "variant": f"{synonym_source} ({threshold})",
                        "synonyms": synonym_dict.find_synonyms(query, threshold),
                    }
                )
    df = pl.from_dicts(rows)
    st.write(df.head(100).to_pandas(), use_container_width=True)

    df = df.with_columns(
        [
            pl.col("synonyms").apply(len).alias("num_synonyms"),
            pl.col("synonyms").apply(lambda synonyms: int(len(synonyms) > 0)).alias("found"),
        ]
    )

    stats_df = (
        df.groupby("variant")
        .agg(
            [
                pl.col("num_synonyms").mean().round(4).alias("num_synonyms_mean"),
                pl.col("num_synonyms").max().alias("num_synonyms_max"),
                pl.col("found").mean().round(4).alias("query_coverage"),
            ]
        )
        .sort("variant")
    )
    st.write(stats_df.to_pandas())


if __name__ == "__main__":
    main()
