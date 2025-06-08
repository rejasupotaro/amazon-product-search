# Data Source

This directory contains scripts and resources for managing the data source used in the Amazon Product Search project. The data is sourced from the [Amazon ESCI Dataset](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset).

## Installation

This package is part of the Amazon Product Search workspace. Install from the root directory:

```shell
# From the root directory
$ uv sync
```

## Download Dataset

To download the dataset, use the provided `Makefile`.

```shell
$ make download
```
