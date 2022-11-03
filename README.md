# Amazon Product Search

This repo showcases and compares various search algorithms and models for [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://github.com/amazon-science/esci-data).

The results of the experiments will be added to the wiki here: https://github.com/rejasupotaro/amazon-product-search/wiki

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

Download the dataset from [this link](https://www.aicrowd.com/challenges/esci-challenge-for-improving-product-search/dataset_files), and run the following command to preprocess the dataset.

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


The ingestion pipeline is built on top of [Apache Beam](https://beam.apache.org/documentation/sdks/python/). If `--encode-text` is given, products are encoded into vectors so that KNN searches can be performed.

The mapping file is located at `schema/es/products.json`. `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color_name`, `product_locale`, and `product_vector` will be indexed.

## Demo

The following command launches [Streamlit](https://streamlit.io/) apps.

```shell
$ docker compose up
$ poetry run inv demo.dataset
$ poetry run inv demo.keyword-extraction
$ poetry run inv demo.search
$ poetry run inv demo.experiment
```

## Experimentation

The demo app provides the ability to run experiments with different experimental settings.

```python
# src/demo/pages/experiment.py
variants=[
    SearchConfig(name="sparse", is_sparse_enabled=True, top_k=100),
    SearchConfig(name="dense", is_dense_enabled=True, top_k=100),
    SearchConfig(name="hybrid", is_sparse_enabled=True, is_dense_enabled=True, top_k=100),
],
```

![](https://user-images.githubusercontent.com/883148/198907715-79f2d99d-59fc-4105-b58f-50e6fd120bf6.png)

## Development

Run the following tasks when you make changes.

```shell
$ poetry run inv lint
$ poetry run pytest -vv
```
