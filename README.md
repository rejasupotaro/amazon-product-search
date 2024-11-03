# Amazon Product Search

![](https://github.com/rejasupotaro/amazon-product-search/actions/workflows/lint_and_test.yml/badge.svg)

This repo showcases and compares various search algorithms and models using [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://github.com/amazon-science/esci-data).

## Project Structure

```
├── .github
│   ├── dependabot.yml
│   └── workflows
│       ├── {project_A}-deploy.yml
│       └── {project_A}-test.yml
├── Makefile       # Common commands
├── pyproject.toml # Common Python configurations
├── README.md
└── src
    ├── {project_A}
    └── {project_B}
```

## Installation

Copy `.envrc.example` and fill in the necessary environment variables. Afterwards, proceed with installing the dependencies.

```shell
$ pyenv install 3.11.8
$ pyenv local 3.11.8
$ pip install poetry
$ poetry env use python
$ poetry install
```

## Development

Run the following tasks after adding any modifications.

```shell
$ make lint
```
