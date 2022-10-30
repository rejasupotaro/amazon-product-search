# Amazon Product Search

This repo is a collection of search algorithms and models for [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://github.com/amazon-science/esci-data).

## Installation

Copy `.envrc.example` and fill in the required environment variables. Then, install the dependencies.

```shell
$ pyenv install 3.9.13
$ pyenv local 3.9.13
$ poetry install
```

The following libraries are necessary to process Japanese.

```shell
# For macOS
$ brew install mecab mecab-ipadic
```

## Dataset

Download the dataset from [here](https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search/dataset_files), and run the following command.

```shell
$ poetry run inv data.split-by-locale
```

## Index Products

This project indexes products to Elasticsearch. If you want to try on your machine, launch Elasticsearch locally and run the document ingestion pipeline against the index you created.

```shell
$ docker compose up
$ poetry run inv es.create_index --locale=jp
$ poetry run inv es.index-docs --locale=jp --es-host=http://localhost:9200 --encode-text --nrows=100
```

The mapping file can be found in `schema/es/products.json`. The ingestion pipeline is built on top of [Apache Beam](https://beam.apache.org/documentation/sdks/python/). If `--encode-text` is given, products are encoded into vectors so that KNN searches can be performed.

## Demo

The following command launches a [Streamlit](https://streamlit.io/) app.

```shell
$ docker compose up
$ poetry run streamlit run src/demo/readme.py
```

## Experimentation

The demo app provides the ability to run experiments with different experimental settings.

```python
# src/demo/pages/experiment.py
variants = [
    SparseSearchConfig(name="title", top_k=100),
    SparseSearchConfig(name="title_description", use_description=True, top_k=100),
    # SparseSearchConfig(name="title_bullet_point", use_bullet_point=True, top_k=100),
    SparseSearchConfig(name="title_brand", use_brand=True, top_k=100),
    # SparseSearchConfig(name="title_color_name", use_color_name=True, top_k=100),
    # DenseSearchConfig(name="dense", top_k=100),
]
```

![](https://user-images.githubusercontent.com/883148/198863889-04ded3bd-3fc0-446a-9bb0-b82b56a5e2bd.png)


## Development

```shell
$ poetry run inv lint
$ poetry run pytest -vv
```
