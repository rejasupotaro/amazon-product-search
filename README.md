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
$ poetry run inv split-dataset-by-locale
$ poetry run inv index --locale=jp
```

## Demo

```shell
$ poetry run streamlit run src/demo/dataset.py
$ poetry run streamlit run src/demo/sparse_search.py
$ poetry run streamlit run src/demo/comparison.py
```
