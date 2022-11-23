FROM docker.io/tensorflow/tensorflow:2.8.0-gpu
# FROM gcr.io/deeplearning-platform-release/tf-cpu.2-8

RUN apt-get update
RUN apt-get install -y \
    libffi-dev libssl-dev zlib1g-dev liblzma-dev tk-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev \
    build-essential git

ENV PYTHON_VERSION 3.10.8
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION
RUN pyenv rehash

ENV PYTHONPATH src:
ENV WORKDIR /app/

WORKDIR $WORKDIR

COPY pyproject.toml poetry.lock $WORKDIR

RUN pip install poetry --no-cache-dir
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-ansi

COPY src .
COPY tasks tasks

RUN pip install -U poetry

COPY google_application_credentials.json .
ENV GOOGLE_APPLICATION_CREDENTIALS google_application_credentials.json

ENTRYPOINT ["poetry", "run"]
