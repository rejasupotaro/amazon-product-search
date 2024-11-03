from typing import Any, Dict


def is_indexable(product: Dict[str, Any]) -> bool:
    """Return True for products to be indexed.

    Args:
        product (dict[str, Any]): A product to assess.

    Returns:
        bool: A flag indicating whether the pipeline will index the doc.
    """
    return bool(product.get("product_title"))
