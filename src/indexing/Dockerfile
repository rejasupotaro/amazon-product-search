ARG PYTHON_VERSION=3.11
ARG BEAM_VERSION=2.49.0

FROM apache/beam_python${PYTHON_VERSION}_sdk:${BEAM_VERSION}

ENV PYTHONPATH src:

WORKDIR /app/src/dense-retrieval

COPY src/dense-retrieval/pyproject.toml pyproject.toml
COPY src/dense-retrieval/poetry.lock poetry.lock
COPY src/dense-retrieval/src src

WORKDIR /app/src/amazon-product-search

COPY src/amazon-product-search/pyproject.toml pyproject.toml
COPY src/amazon-product-search/poetry.lock poetry.lock
COPY src/amazon-product-search/src src

WORKDIR /app/src/indexing

COPY src/indexing/pyproject.toml pyproject.toml
COPY src/indexing/poetry.lock poetry.lock
COPY src/indexing/src src

RUN pip install --upgrade pip && \
    pip install -U poetry --no-cache-dir
RUN poetry config virtualenvs.create false && \
    poetry install --without dev --no-interaction --no-ansi
RUN python -m unidic download
