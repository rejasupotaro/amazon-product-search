from typing import Any


def _get_search_fields(
    use_description: bool, use_bullet_point: bool, use_brand: bool, use_color_name: bool
) -> list[str]:
    fields = ["product_title"]
    if use_description:
        fields.append("product_description")
    if use_bullet_point:
        fields.append("product_bullet_point")
    if use_brand:
        fields.append("product_brand")
    if use_color_name:
        fields.append("product_color_name")
    return fields


def build_multimatch_search_query(
    query: str,
    use_description: bool = False,
    use_bullet_point: bool = False,
    use_brand: bool = False,
    use_color_name: bool = False,
) -> dict[str, Any]:
    """Build a multimatch ES query.

    Returns:
        dict[str, Any]: The constructed ES query.
    """
    if not query:
        return {
            "match_all": {},
        }

    return {
        "multi_match": {
            "query": query,
            "fields": _get_search_fields(use_description, use_bullet_point, use_brand, use_color_name),
            "operator": "or",
        }
    }


def build_knn_search_query(query_vector: list[float], top_k: int) -> dict[str, Any]:
    """Build a KNN ES query from given conditions.

    Args:
        query_vector (list[float]): An encoded query vector.
        top_k (int): A number specifying how many results to return.

    Returns:
        dict[str, Any]: The constructed ES query.
    """
    return {
        "field": "product_vector",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 100,
    }
