import pytest

from amazon_product_search.core.retrieval.score_normalizer import min_max_scale


@pytest.mark.parametrize(
    ("x", "min_val", "max_val", "expected"),
    [
        ([], None, None, []),
        ([1], None, None, [0.5]),
        ([1, 2, 3], None, None, [0, 0.5, 1]),
        ([3, 2, 1], None, None, [1, 0.5, 0]),
        ([1, 2, 3], 1, 3, [0, 0.5, 1]),
        ([1, 2, 3], 0, 2, [0.5, 1, 1.5]),
    ],
)
def test_min_max_scale(x, min_val, max_val, expected):
    actual = min_max_scale(x, min_val, max_val)
    assert actual == expected
