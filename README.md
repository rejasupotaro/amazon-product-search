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
$ poetry run inv data.split--y-locale
$ poetry run inv es.create_index --locale=jp
$ poetry run inv es.index_doc --locale=jp
```

## Demo

```shell
$ poetry run streamlit run src/demo/readme.py
```
