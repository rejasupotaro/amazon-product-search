# Amazon Product Search

![](https://github.com/rejasupotaro/amazon-product-search/actions/workflows/lint_and_test.yml/badge.svg)

This repo showcases and compares various search algorithms and models using [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://github.com/amazon-science/esci-data).

The details and results of experiments can be found in this wiki: https://github.com/rejasupotaro/amazon-product-search/wiki

## Installation

Copy `.envrc.example` and fill in the necessary environment variables. Afterwards, proceed with installing the dependencies.

```shell
$ pyenv install 3.10.8
$ pyenv local 3.10.8
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

## Index Products

This project involves indexing products into Elasticsearch. If you want to try it on your own machine, launch Elasticsearch locally and execute the document indexing pipeline against the created index.

```shell
$ docker compose up
$ poetry run inv es.create-index --index-name=products_jp
$ poetry run inv es.index \
  --index-name=products_jp \
  --locale=jp \
  --dest-host=http://localhost:9200 \
  --extract-keywords \
  --encode-text \
  --nrows=100
```

See https://github.com/rejasupotaro/amazon-product-search/wiki/Indexing for more details.

## Demo

The command below launches the [Streamlit](https://streamlit.io/) demo app.

```shell
# Launch Elasticsearch beforehand
$ docker compose up

$ poetry run inv demo.search
```

![](https://user-images.githubusercontent.com/883148/203654537-8b495c9c-f8af-4c3f-90f9-60edacf647b9.png)

## Experimentation

The demo application allows for the execution of experiments with different experimental settings.

```python
# src/demo/experimental_setup.py
"sparse_vs_dense": ExperimentalSetup(
    index_name="products_jp",
    locale="jp",
    num_queries=5000,
    variants=[
        Variant(name="sparse", fields=["product_title"]),
        Variant(name="dense", fields=["product_vector"]),
        Variant(name="hybrid", fields=["product_title", "product_vector"]),
    ],
),
```

![](https://user-images.githubusercontent.com/883148/199724869-f8c51c10-da16-42de-a2fe-bf112864c083.png)

## Development

Run the following tasks after adding any modifications.

```shell
$ poetry run black .
$ poetry run ruff . --fix
$ poetry run mypy src
$ poetry run pytest tests/unit -vv
$ poetry run pytest tests/integration -vv
```
