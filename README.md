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

Download the dataset from https://github.com/amazon-science/esci-data and put them in `data/raw`. Then, run the following command to preprocess the dataset.

```shell
$ poetry run inv data.merge-and-split
```

## Index Products

This project indexes products to Elasticsearch. If you want to try on your machine, launch Elasticsearch locally and run the document ingestion pipeline against the index you created.

```shell
$ docker compose up
$ poetry run inv es.create_index --index-name=products_jp
$ poetry run inv es.index-docs \
  --index-name=products_jp \
  --locale=jp \
  --es-host=http://localhost:9200 \
  --extract-keywords \
  --encode-text \
  --nrows=100
```


The ingestion pipeline is built on top of [Apache Beam](https://beam.apache.org/documentation/sdks/python/). If `--encode-text` is given, products are encoded into vectors so that KNN searches can be performed.

The mapping file is located at `schema/es/products.json`. `product_id`, `product_title`, `product_description`, `product_bullet_point`, `product_brand`, `product_color`, `product_locale`, and `product_vector` will be indexed.

## Demo

The following command launches the [Streamlit](https://streamlit.io/) demo app.

```shell
$ docker compose up
$ poetry run inv demo
```

## Experimentation

The demo app provides the ability to run experiments with different experimental settings.

```python
# src/demo/experimental_setup.py
"sparse_vs_dense": ExperimentalSetup(
    index_name="products_jp",
    locale="jp",
    num_queries=5000,
    variants=[
        Variant(name="sparse", fields=["product_title"]),
        Variant(name="dense", fields=["product_vector"]),
        Variant(name="hybrid", fields=["product_title", "product_vector"]),
    ],
),
```

![](https://user-images.githubusercontent.com/883148/199724869-f8c51c10-da16-42de-a2fe-bf112864c083.png)

## Development

Run the following tasks when you make changes.

```shell
$ poetry run inv format lint
$ poetry run pytest -vv
```
