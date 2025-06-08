# Core - Amazon Product Search

## Installation

This package is part of the Amazon Product Search workspace. Install from the root directory:

```shell
# From the root directory
$ uv sync
```

The following libraries are necessary for Japanese text processing:

```shell
# For macOS
$ brew install mecab mecab-ipadic

$ uv run python -m unidic download
```

## Dataset

Clone https://github.com/amazon-science/esci-data and copy `esci-data/shopping_queries_dataset/*` into `amazon-product/search/data/raw/`. Then, run the following command to preprocess the dataset.

```shell
$ uv run --package amazon-product-search inv data.merge-and-split
```
