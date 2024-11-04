FROM gcr.io/deeplearning-platform-release/pytorch-gpu

RUN apt-get update && \
    apt-get install -y \
    libffi-dev libssl-dev zlib1g-dev liblzma-dev tk-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev \
    build-essential git

ENV PYTHON_VERSION 3.11.8
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash

WORKDIR /app/src/dense-retrieval

COPY src/dense-retrieval/pyproject.toml pyproject.toml
COPY src/dense-retrieval/poetry.lock poetry.lock
COPY src/dense-retrieval/src src

WORKDIR /app/src/amazon-product-search

COPY src/amazon-product-search/pyproject.toml pyproject.toml
COPY src/amazon-product-search/poetry.lock poetry.lock
COPY src/amazon-product-search/src src

WORKDIR /app/src/training

COPY src/training/pyproject.toml pyproject.toml
COPY src/training/poetry.lock poetry.lock
COPY src/training/src src
COPY src/training/Makefile Makefile

RUN pip install poetry --no-cache-dir && \
    poetry config virtualenvs.create false && \
    poetry install --only main

COPY gcloud/training.json training.json
ENV GOOGLE_APPLICATION_CREDENTIALS training.json

ENTRYPOINT ["make", "run"]
