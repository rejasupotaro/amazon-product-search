# Amazon Product Search

This repo is a collection of models and demos for Amazon Product Search.

## Installation

Install the appropriate Python and dependencies.

```
$ pyenv install 3.9.13
$ pyenv local 3.9.13
$ poetry install
```

## Dataset

Download the dataset on the internet. Then run the following command.

```
$ poetry run inv split-dataset-by-locale
```

## Demo

```
$ poetry run streamlit run src/demo/catalog.py
```
