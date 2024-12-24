import pytest

from amazon_product_search.synonyms.filters.utils import (
    are_two_sets_identical,
    is_either_contained_in_other,
    is_either_number,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("ab", "ab", True),
        ("ab", "cd", False),
        ("ab cd", "cd ab", True),
    ],
)
def test_are_two_sets_identical(a, b, expected):
    assert are_two_sets_identical(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("ab", "abcd", True),
        ("ab", "cd", False),
        ("abcd", "ab", True),
    ],
)
def test_is_either_contained_in_other(a, b, expected):
    assert is_either_contained_in_other(a, b) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("abc", "def", False),
        ("123", "abc", True),
        ("abc", "123", True),
        ("123", "123", True),
    ],
)
def test_is_either_number(a, b, expected):
    assert is_either_number(a, b) == expected
