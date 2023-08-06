import numpy as np
import pytest

from amazon_product_search.core.retrieval.score_normalizer import min_max_scale


@pytest.mark.parametrize(
    ("x", "min_val", "max_val", "expected"),
    [
        (np.array([1, 2, 3]), None, None, np.array([0, 0.5, 1])),
        (np.array([1, 2, 3]), 1, 3, np.array([0, 0.5, 1])),
        (np.array([1, 2, 3]), 0, 2, np.array([0.5, 1, 1.5])),
    ],
)
def test_min_max_scale(x, min_val, max_val, expected):
    actual = min_max_scale(x, min_val, max_val)
    print(actual)
    assert actual == pytest.approx(expected)
