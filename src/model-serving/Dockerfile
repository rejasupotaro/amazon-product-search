FROM nvcr.io/nvidia/tritonserver:24.01-py3

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --timeout 2000 -r /tmp/requirements.txt

