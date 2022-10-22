from typing import Any


def build_search_query(query, use_description, use_bullet_point, use_brand, use_color_name) -> dict[str, Any]:
    if not query:
        return {
            "match_all": {},
        }

    fields = ["product_title"]
    if use_description:
        fields.append("product_description")
    if use_bullet_point:
        fields.append("product_bullet_point")
    if use_brand:
        fields.append("product_brand")
    if use_color_name:
        fields.append("product_color_name")

    return {
        "multi_match": {
            "query": query,
            "fields": fields,
            "operator": "or",
        }
    }


def build_knn_search_query(query_vector: list[float], top_k: int) -> dict[str, Any]:
    return {
        "field": "product_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 100,
    }
