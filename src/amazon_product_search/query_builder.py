from typing import Any, Dict

from amazon_product_search.models.search import RequestParams


def build(params: RequestParams) -> Dict[str, Any]:
    if not params.query:
        return {
            "match_all": {},
        }

    fields = ["product_title"]
    if params.use_description:
        fields.append("product_description")
    if params.use_bullet_point:
        fields.append("product_bullet_point")
    if params.use_brand:
        fields.append("product_brand")
    if params.use_color_name:
        fields.append("product_color_name")

    return {
        "multi_match": {
            "query": params.query,
            "fields": fields,
            "operator": "or",
        }
    }
