# Indexing - Amazon Product Search

## Installation

This package is part of the Amazon Product Search workspace. Install from the root directory:

### Option 1: UV (Recommended - Fast & Modern)

```shell
# From the root directory
$ uv sync
```

### Option 2: Poetry (Backward Compatibility)

```shell
$ pyenv install 3.11.8
$ pyenv local 3.11.8
$ pip install poetry
$ poetry env use python
$ poetry install
```

The following libraries are necessary for Japanese text processing:

```shell
# For macOS
$ brew install mecab mecab-ipadic

# UV (recommended)
$ uv run python -m unidic download

# Poetry (alternative)
$ poetry run python -m unidic download
```

## Index Products

This project involves indexing products into search engines. If you'd like to test it on your own machine, you can start by launching Elasticsearch or Vespa locally. Then, execute the document indexing pipeline against the created index.

```shell
$ docker compose --profile elasticsearch up

# UV (recommended)
$ uv run --package amazon-product-search inv es.create-index --index-name=products_jp
$ uv run --package indexing inv indexing.feed \
  --index-name=products_jp \
  --locale=jp \
  --dest=es \
  --dest-host=http://localhost:9200 \
  --nrows=10

# Poetry (alternative)
$ poetry run inv es.create-index --index-name=products_jp
$ poetry run inv indexing.feed \
  --index-name=products_jp \
  --locale=jp \
  --dest=es \
  --dest-host=http://localhost:9200 \
  --nrows=10
```
