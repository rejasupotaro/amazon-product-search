# Amazon Product Search

![](https://github.com/rejasupotaro/amazon-product-search/actions/workflows/lint_and_test.yml/badge.svg)

This repo showcases and compares various search algorithms and models using [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://github.com/amazon-science/esci-data).

## Installation

Copy `.envrc.example` and fill in the necessary environment variables. Afterwards, proceed with installing the dependencies.

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

## Index Products

This project involves indexing products into search engines. If you'd like to test it on your own machine, you can start by launching Elasticsearch or Vespa locally. Then, execute the document indexing pipeline against the created index.

```shell
$ docker compose --profile elasticsearch up
$ poetry run inv es.create-index --index-name=products_jp
$ poetry run inv indexing.feed \
  --index-name=products_jp \
  --locale=jp \
  --dest=es \
  --dest-host=http://localhost:9200 \
  --nrows=10
```

## Demo

The command below launches the [Streamlit](https://streamlit.io/) demo app.

```shell
# Launch Elasticsearch beforehand
$ docker compose --profile elasticsearch up

$ poetry run inv demo.es
```

![](https://user-images.githubusercontent.com/883148/203654537-8b495c9c-f8af-4c3f-90f9-60edacf647b9.png)

## Development

Run the following tasks after adding any modifications.

```shell
$ poetry run black .
$ poetry run ruff . --fix
$ poetry run mypy src
$ poetry run pytest tests/unit -vv
$ poetry run pytest tests/integration -vv
```
