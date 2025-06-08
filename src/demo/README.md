# Demo - Amazon Product Search

## Installation

This package is part of the Amazon Product Search workspace. Install from the root directory:

```shell
# From the root directory
$ uv sync
```

## Demo

The command below launches the [Streamlit](https://streamlit.io/) demo app.

```shell
# Using make (runs with available Python environment)
$ make run_eda
$ make run_tokenization
$ make run_es
$ make run_vespa

# Or run directly with UV
$ uv run --package demo streamlit run src/demo/apps/eda/📊_Product_Catalogue.py
$ uv run --package demo streamlit run src/demo/apps/features/🤖_AutoTokenizer.py
$ uv run --package demo streamlit run src/demo/apps/es/🔍_Retrieval.py
$ uv run --package demo streamlit run src/demo/apps/vespa/🔍_Retrieval.py
```

![](https://user-images.githubusercontent.com/883148/203654537-8b495c9c-f8af-4c3f-90f9-60edacf647b9.png)
