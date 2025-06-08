# Dense Retrieval - Amazon Product Search

## Installation

This package is part of the Amazon Product Search workspace. Install from the root directory:

```shell
# From the root directory
$ uv sync
```

The following libraries are necessary for Japanese text processing:

```shell
# For macOS
$ brew install mecab mecab-ipadic

$ uv run python -m unidic download
```
