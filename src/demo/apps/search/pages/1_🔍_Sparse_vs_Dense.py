import streamlit as st

from amazon_product_search.es.es_client import EsClient
from amazon_product_search.es.query_builder import QueryBuilder
from amazon_product_search.metrics import compute_ndcg, compute_recall
from amazon_product_search.nlp.normalizer import normalize_query
from demo.apps.search.search_ui import draw_products
from demo.page_config import set_page_config
from demo.utils import load_merged

es_client = EsClient()
query_builder = QueryBuilder()


@st.cache_data
def load_dataset() -> dict[str, dict[str, tuple[str, str]]]:
    df = load_merged(locale="jp").to_pandas()
    df = df[df["split"] == "test"]
    query_to_label: dict[str, dict[str, tuple[str, str]]] = {}
    for query, group in df.groupby("query"):
        query_to_label[query] = {}
        for row in group.to_dict("records"):
            query_to_label[query][row["product_id"]] = (row["esci_label"], row["product_title"])
    return query_to_label


def main():
    set_page_config()
    st.write("## Search")

    queries, query_to_label = None, {}
    use_dataset = st.checkbox("Use Dataset:", value=True)
    if use_dataset:
        query_to_label = load_dataset()
        queries = query_to_label.keys()

    st.write("#### Input")
    with st.form("input"):
        index_name = st.selectbox("Index:", es_client.list_indices())

        query = st.selectbox("Query:", queries) if queries else st.text_input("Query:")

        sparse_fields = st.multiselect(
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

        query_type = st.selectbox(
            "Query Type:",
            options=["combined_fields", "cross_fields", "best_fields", "simple_query_string"],
        )

        is_synonym_expansion_enabled = st.checkbox("enable_synonym_expansion")

        if not st.form_submit_button("Search"):
            return

    size = 20

    normalized_query = normalize_query(query)
    sparse_query = query_builder.build_sparse_search_query(
        query=normalized_query,
        fields=sparse_fields,
        query_type=query_type,
        is_synonym_expansion_enabled=is_synonym_expansion_enabled,
    )

    query_vector = query_builder.encode(normalized_query)
    dense_query = {
        "query_vector": query_vector,
        "field": "product_vector",
        "k": size,
        "num_candidates": size,
    }

    st.write("----")

    label_dict = query_to_label.get(query, {})
    if label_dict:
        with st.expander("Labels", expanded=False):
            st.write(label_dict)

    sparse_response = es_client.search(index_name=index_name, query=sparse_query, size=size)
    dense_response = es_client.knn_search(index_name=index_name, knn_query=dense_query)

    sparse_retrieved_ids = [result.product["product_id"] for result in sparse_response.results]
    dense_retrieved_ids = [result.product["product_id"] for result in dense_response.results]
    union = set(sparse_retrieved_ids) | set(dense_retrieved_ids)
    intersection = set(sparse_retrieved_ids) & set(dense_retrieved_ids)
    iou = round(len(intersection) / len(union), 4)

    st.write("----")

    st.write("#### Output")

    st.write(f"IoU: {iou}")

    columns = st.columns(2)
    judgements = {product_id: label for product_id, (label, product_title) in label_dict.items()}
    relevant_ids = {product_id for product_id, (label, product_title) in label_dict.items() if label == "E"}
    with columns[0]:
        header = f"{sparse_response.total_hits} products found"
        if label_dict:
            ndcg = compute_ndcg(sparse_retrieved_ids, judgements)
            recall = compute_recall(sparse_retrieved_ids, relevant_ids)
            header = f"{header} (NDCG: {ndcg}, Recall: {recall})"
        st.write(header)
        draw_products(sparse_response.results, label_dict)
    with columns[1]:
        header = f"{dense_response.total_hits} products found"
        if label_dict:
            ndcg = compute_ndcg(dense_retrieved_ids, judgements)
            recall = compute_recall(dense_retrieved_ids, relevant_ids)
            header = f"{header} (NDCG: {ndcg}, Recall: {recall})"
        st.write(header)
        draw_products(dense_response.results, label_dict)


if __name__ == "__main__":
    main()
