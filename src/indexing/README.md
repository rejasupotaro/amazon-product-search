# Indexing - Amazon Product Search

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
