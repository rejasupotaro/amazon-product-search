import streamlit as st

from amazon_product_search.constants import HF
from amazon_product_search.core import source
from amazon_product_search.core.es.es_client import EsClient
from amazon_product_search.core.es.query_builder import QueryBuilder
from amazon_product_search.core.metrics import (
    compute_iou,
    compute_ndcg,
    compute_precision,
    compute_recall,
)
from amazon_product_search.core.nlp.normalizer import normalize_query
from amazon_product_search.core.retrieval.response import Response
from amazon_product_search.core.source import Locale
from demo.apps.es.search_ui import draw_products
from demo.page_config import set_page_config

es_client = EsClient()


@st.cache_resource
def get_query_builder(locale: Locale) -> QueryBuilder:
    hf_model_name = HF.LOCALE_TO_MODEL_NAME[locale]
    return QueryBuilder(locale=locale, hf_model_name=hf_model_name)


@st.cache_data
def load_dataset(locale: Locale) -> dict[str, dict[str, tuple[str, str]]]:
    df = source.load_merged(locale).to_pandas()
    df = df[df["split"] == "test"]
    query_to_label: dict[str, dict[str, tuple[str, str]]] = {}
    for query, group in df.groupby("query"):
        query_to_label[query] = {}
        for row in group.to_dict("records"):
            query_to_label[query][row["product_id"]] = (
                row["esci_label"],
                row["product_title"],
            )
    return query_to_label


def search(
    locale: Locale,
    index_name: str,
    query: str,
    lexical_fields: list[str],
    enable_synonym_expansion: bool,
    size: int = 100,
) -> tuple[Response, Response]:
    normalized_query = normalize_query(query)
    lexical_query = get_query_builder(locale).build_lexical_search_query(
        query=normalized_query,
        fields=lexical_fields,
        enable_synonym_expansion=enable_synonym_expansion,
    )

    query_vector = get_query_builder(locale).encode(normalized_query)
    semantic_query = {
        "query_vector": query_vector,
        "field": "product_vector",
        "k": size,
        "num_candidates": size,
    }

    lexical_response = es_client.search(index_name=index_name, query=lexical_query, size=size)
    semantic_response = es_client.search(index_name=index_name, knn_query=semantic_query)

    return lexical_response, semantic_response


def main() -> None:
    set_page_config()
    st.write("## Search")

    with st.sidebar:
        locale = st.selectbox("Locale", ["us", "jp"])
        index_name = str(st.selectbox("Index:", es_client.list_indices()))
        queries, query_to_label = None, {}
        use_dataset = st.checkbox("Use Dataset:", value=True)
        if use_dataset:
            query_to_label = load_dataset(locale)
            queries = query_to_label.keys()

    st.write("#### Input")
    with st.form("input"):
        query = st.selectbox("Query:", queries) if queries else st.text_input("Query:")
        assert query is not None

        lexical_fields = st.multiselect(
            "Fields:",
            options=[
                "product_title",
                "product_description",
                "product_bullet_point",
                "product_brand",
                "product_color",
            ],
            default=["product_title"],
        )

        enable_synonym_expansion = st.checkbox("enable_synonym_expansion")

        if not st.form_submit_button("Search"):
            return

    st.write("----")

    label_dict = query_to_label.get(query, {})
    if label_dict:
        with st.expander("Labels", expanded=False):
            st.write(label_dict)

    st.write("----")

    st.write("#### Output")

    relevant_ids = {product_id for product_id, (label, product_title) in label_dict.items() if label == "E"}
    lexical_response, semantic_response = search(locale, index_name, query, lexical_fields, enable_synonym_expansion)
    lexical_retrieved_ids = [result.product["product_id"] for result in lexical_response.results]
    semantic_retrieved_ids = [result.product["product_id"] for result in semantic_response.results]
    lexical_relevant_ids = [retrieved_id for retrieved_id in lexical_retrieved_ids if retrieved_id in relevant_ids]
    semantic_relevant_ids = [retrieved_id for retrieved_id in semantic_retrieved_ids if retrieved_id in relevant_ids]

    all_iou, all_intersection, all_union = compute_iou(set(lexical_retrieved_ids), set(semantic_retrieved_ids))
    all_iou_text = f"All IoU: {all_iou} ({len(all_intersection)} / {len(all_union)})"
    relevant_iou, relevant_intersection, relevant_union = compute_iou(
        set(lexical_relevant_ids), set(semantic_relevant_ids)
    )
    relevant_iou_text = f"Relevant IoU: {relevant_iou} ({len(relevant_intersection)} / {len(relevant_union)})"
    st.write(f"{all_iou_text}, {relevant_iou_text}")

    columns = st.columns(2)
    id_to_label = {product_id: label for product_id, (label, product_title) in label_dict.items()}
    with columns[0]:
        header = f"{lexical_response.total_hits} products found"
        if label_dict:
            precision = compute_precision(lexical_retrieved_ids, relevant_ids)
            recall = compute_recall(lexical_retrieved_ids, relevant_ids)
            ndcg = compute_ndcg(lexical_retrieved_ids, id_to_label)
            header = f"{header} (Precision: {precision}, Recall: {recall}, NDCG: {ndcg})"
        st.write(header)
        draw_products(lexical_response.results, label_dict)
    with columns[1]:
        header = f"{semantic_response.total_hits} products found"
        if label_dict:
            precision = compute_precision(semantic_retrieved_ids, relevant_ids)
            recall = compute_recall(semantic_retrieved_ids, relevant_ids)
            ndcg = compute_ndcg(semantic_retrieved_ids, id_to_label)
            header = f"{header} (Preicison: {precision}, Recall: {recall}, NDCG: {ndcg})"
        st.write(header)
        draw_products(semantic_response.results, label_dict)


if __name__ == "__main__":
    main()
