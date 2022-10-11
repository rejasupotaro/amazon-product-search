from typing import Any, Dict

from amazon_product_search.models.search import RequestParams


def build(params: RequestParams) -> Dict[str, Any]:
    if not params.query:
        return {
            "match_all": {},
        }

    if params.use_description:
        return {
            "bool": {
                "minimum_should_match": 1,
                "should": [
                    {
                        "match": {
                            "product_title": {
                                "query": params.query,
                            },
                        }
                    },
                    {
                        "match": {
                            "product_description": {
                                "query": params.query,
                            },
                        }
                    },
                ],
            },
        }

    return {
        "match": {
            "product_title": {
                "query": params.query,
            },
        },
    }
