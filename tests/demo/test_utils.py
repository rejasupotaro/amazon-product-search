from demo.utils import split_fields


def test_split_fields():
    fields = ["title", "description", "vector"]
    sparse_fields, dense_fields = split_fields(fields)
    assert sparse_fields == ["title", "description"]
    assert dense_fields == ["vector"]
