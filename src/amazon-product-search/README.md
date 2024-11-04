# Core - Amazon Product Search

## Installation

```shell
$ pyenv install 3.11.8
$ pyenv local 3.11.8
$ pip install poetry
$ poetry env use python
$ poetry install
```

The following libraries are necessary for Japanese text processing.

```shell
# For macOS
$ brew install mecab mecab-ipadic
$ poetry run python -m unidic download
```

## Dataset

Clone https://github.com/amazon-science/esci-data and copy `esci-data/shopping_queries_dataset/*` into `amazon-product/search/data/raw/`. Then, run the following command to preprocess the dataset.

```shell
$ poetry run inv data.merge-and-split
```
