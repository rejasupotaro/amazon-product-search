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

    return {
        "multi_match": {
            "query": params.query,
            "fields": fields,
            "operator": "or",
        }
    }
