from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from amazon_product_search import query_builder, source
from amazon_product_search.es_client import EsClient
from amazon_product_search.metrics import compute_ap, compute_ndcg
from amazon_product_search.models.search import RequestParams, Response, Result

es_client = EsClient(
    es_host="http://localhost:9200",
)


@dataclass
class Variant:
    name: str
    use_description: bool = False
    use_bullet_point: bool = False
    use_brand: bool = False
    use_color_name: bool = False
    top_k: int = False


@st.cache
def load_labels(locale: str, nrows: int) -> pd.DataFrame:
    return source.load_labels(locale, nrows)


def compute_metrics(variant: Variant, query: str, labels_df: pd.DataFrame) -> Dict[str, Any]:
    params = RequestParams(
        query=query,
        use_description=variant.use_description,
        use_bullet_point=variant.use_bullet_point,
        use_brand=variant.use_brand,
        use_color_name=variant.use_color_name,
    )
    es_query = query_builder.build(params)

    es_response = es_client.search(index_name="products_jp", es_query=es_query, size=variant.top_k)

    response = Response(
        results=[Result(product=hit["_source"], score=hit["_score"]) for hit in es_response["hits"]["hits"]],
        total_hits=es_response["hits"]["total"]["value"],
    )
    retrieved_ids = [result.product["product_id"] for result in response.results]
    relevant_ids = labels_df[labels_df["esci_label"] == "exact"]["product_id"].tolist()
    judgements: Dict[str, str] = {row["product_id"]: row["esci_label"] for row in labels_df.to_dict("records")}
    return {
        "variant": variant.name,
        "query": query,
        "total_hits": response.total_hits,
        "map": compute_ap(retrieved_ids, relevant_ids),
        "ndcg": compute_ndcg(retrieved_ids, judgements),
    }


def main():
    st.set_page_config(page_icon="️🔍", layout="wide")

    st.write("## Offline Experiment")

    locale = "jp"
    nrows = 10000

    variants = [
        Variant(name="title", top_k=100),
        Variant(name="title_description", use_description=True, top_k=100),
        Variant(name="title_bullet_point", use_bullet_point=True, top_k=100),
        Variant(name="title_brand", use_brand=True, top_k=100),
        Variant(name="title_color_name", use_color_name=True, top_k=100),
    ]
    st.write("#### Variants")
    st.write(variants)

    clicked = st.button("Run")

    if not clicked:
        return

    labels_df = load_labels(locale, nrows)
    query_dict: Dict[str, pd.DataFrame] = {}
    for query, query_labels_df in labels_df.groupby("query"):
        query_dict[query] = query_labels_df

    total_examples = len(variants) * len(query_dict)
    n = 0
    progress_text = st.empty()
    progress_bar = st.progress(0)
    metrics = []
    for variant in variants:
        for query, query_labels_df in query_dict.items():
            progress_text.text(f"Query ({n} / {total_examples}): {query}")
            n += 1
            progress_bar.progress(n / total_examples)
            metrics.append(compute_metrics(variant, query, query_labels_df))
    progress_text.text(f"Done ({n} / {total_examples})")

    st.write("#### Result")
    metrics_df = pd.DataFrame(metrics)
    stats_df = (
        metrics_df.groupby("variant")
        .agg(
            {
                "total_hits": "mean",
                "map": "mean",
                "ndcg": "mean",
            }
        )
        .reset_index()
    )
    stats_df["total_hits"] = stats_df["total_hits"].apply(int)
    st.write(stats_df)

    for metric in ["total_hits", "map", "ndcg"]:
        fig = px.box(metrics_df, y=metric, color="variant")
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
