# Dense Retrieval - Amazon Product Search

## Installation

This package is part of the Amazon Product Search workspace. Install from the root directory:

### Option 1: UV (Recommended - Fast & Modern)

```shell
# From the root directory
$ uv sync
```

### Option 2: Poetry (Backward Compatibility)

```shell
$ pyenv install 3.11.8
$ pyenv local 3.11.8
$ pip install poetry
$ poetry env use python
$ poetry install
```

The following libraries are necessary for Japanese text processing:

```shell
# For macOS
$ brew install mecab mecab-ipadic

# UV (recommended)
$ uv run python -m unidic download

# Poetry (alternative)
$ poetry run python -m unidic download
```
