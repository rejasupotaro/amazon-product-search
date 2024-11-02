from typing import Any, Dict


def is_indexable(product: Dict[str, Any]) -> bool:
    """Return True for products to be indexed.

    Args:
        product (dict[str, Any]): A product to assess.

    Returns:
        bool: A flag indicating whether the pipeline will index the doc.
    """
    if product is None:
        return False
    if "product_title" not in product:
        return False
    if not product["product_title"]:
        return False
    return True
