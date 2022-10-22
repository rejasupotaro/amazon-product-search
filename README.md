# Amazon Product Search

This repo is a collection of models and demos for Amazon Product Search.

## Installation

Install the appropriate Python and dependencies.

```shell
$ pyenv install 3.9.13
$ pyenv local 3.9.13
$ poetry install
```

```shell
# For macOS
$ brew install mecab mecab-ipadic
```

## Dataset

Download the dataset on the internet. Then run the following command.

```shell
$ poetry run inv data.split--by-locale
$ poetry run inv es.create_index --locale=jp
$ poetry run inv es.index_doc --locale=jp
```

## Demo

```shell
$ poetry run streamlit run src/demo/readme.py
```
