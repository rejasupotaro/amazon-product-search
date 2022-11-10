import pandas as pd
import streamlit as st

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.response import Result
from amazon_product_search.metrics import compute_ndcg
from amazon_product_search.reranking.reranker import RandomReranker, SentenceBERTReranker
from demo.page_config import set_page_config
from demo.utils import load_merged

es_client = EsClient()
random_reranker = RandomReranker()
sbert_reranker = SentenceBERTReranker()


@st.cache
def extract_judgements(df: pd.DataFrame) -> dict[str, str]:
    return {row["product_id"]: row["esci_label"] for row in df[["product_id", "esci_label"]].to_dict("records")}


def search(query: str, doc_ids: list[str], all_judgements: dict[str, str]) -> list[Result]:
    doc_id_filter_clauses = [{"term": {"product_id.keyword": doc_id}} for doc_id in doc_ids]
    es_query = {
        "bool": {
            "filter": [
                {
                    "bool": {
                        "should": doc_id_filter_clauses,
                    }
                }
            ],
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["product_title", "product_bullet_point"],
                        "operator": "or",
                    }
                }
            ],
        }
    }
    response = es_client.search(index_name="products_all_jp", query=es_query)
    for result in response.results:
        result.product["query"] = query
        result.product["esci_label"] = all_judgements[result.product["product_id"]]
    return response.results


def draw_results(results: list[Result]):
    products = [result.product for result in results]

    retrieved_ids = [product["product_id"] for product in products]
    judgements: dict[str, str] = {product["product_id"]: product["esci_label"] for product in products}
    ndcg = compute_ndcg(retrieved_ids, judgements)
    if ndcg:
        ndcg = round(ndcg, 4)
    st.write(f"NDCG: {ndcg}")

    df = pd.DataFrame(products)
    st.write(df[["esci_label", "query", "product_title", "product_id"]])


def main():
    set_page_config()

    st.write("## Reranking Experiments")
    df = load_merged(locale="jp")
    all_judgements = extract_judgements(df)

    queries = df["query"].unique()
    query = st.selectbox("Query:", queries)
    filtered_df = df[df["query"] == query]

    products = filtered_df.to_dict("records")
    results = [Result(product=product, score=1) for product in products]

    columns = st.columns(3)
    with columns[0]:
        st.write("### Random Results")
        random_results = random_reranker.rerank(query, results)
        draw_results(random_results)
    with columns[1]:
        st.write("### Search Results")
        search_results = search(
            query, doc_ids=[result.product["product_id"] for result in results], all_judgements=all_judgements
        )
        draw_results(search_results)
    with columns[2]:
        st.write("### SBERT Results")
        sbert_results = sbert_reranker.rerank(query, results)
        draw_results(sbert_results)


if __name__ == "__main__":
    main()
