# Triton Inference Server

This directory contains a text embedding model and its server.

## Export Model

Before launching the server, convert a text embedding model into the ONNX format. You can achieve this by executing the following command at the project root.

```
poetry run inv model.export
```

Then, move the exported ONNX model to `serving/text_embedding/model_repository/all_minilm/1/model.onnx`.

## Launch Server

To launch the Triton Inference Server, execute the command `docker compose up`.
