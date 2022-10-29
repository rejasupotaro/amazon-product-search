# Amazon Product Search

This repo is a collection of search algorithms and models for Amazon Product Search.

## Installation

First, create `.envrc` and fill in the required environment variables. Then, install the appropriate Python version and dependencies.

```shell
$ pyenv install 3.9.13
$ pyenv local 3.9.13
$ poetry install
```

To process Japanese, install the following libraries.

```shell
# For macOS
$ brew install mecab mecab-ipadic
```

## Dataset

Download the dataset on the internet. Then run the following command.

```shell
$ poetry run inv data.split-by-locale
$ poetry run inv es.create_index --locale=jp
```

## Index Documents

```shell
$ docker compose up
$ poetry run inv es.index-docs --locale=jp --es-host=http://localhost:9200 --encode-text --nrows=100
```

## Demo

```shell
$ docker compose up
$ poetry run streamlit run src/demo/readme.py
```

## Development

```shell
$ poetry run inv lint
$ poetry run pytest -vv
```
