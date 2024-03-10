# Triton Inference Server

This directory contains a text embedding model and its server.

## Export Model

Before launching the server, convert a text embedding model into the ONNX format. You can achieve this by executing the following command at the project root.

```
poetry run inv model.export
```

Then, move the exported ONNX model to `serving/text_embedding/model_repository/all_minilm/1/model.onnx`.

## Launch Server

To launch the Triton Inference Server, execute the following commands:

```
docker compose build
docker compose up
```

It launches the following services:

```
curl http://localhost:8000/v2/models/tokenizer
{"name":"tokenizer","versions":["1"],"platform":"python","inputs":[{"name":"text","datatype":"BYTES","shape":[-1]}],"outputs":[{"name":"input_ids","datatype":"INT64","shape":[-1,512]},{"name":"attention_mask","datatype":"INT64","shape":[-1,512]}]}

curl http://localhost:8000/v2/models/all_minilm
{"name":"all_minilm","versions":["1"],"platform":"onnxruntime_onnx","inputs":[{"name":"input_ids","datatype":"INT64","shape":[-1,512]},{"name":"attention_mask","datatype":"INT64","shape":[-1,512]}],"outputs":[{"name":"output","datatype":"FP32","shape":[-1,384]}]}

http://localhost:8000/v2/models/text_embedding
{"name":"text_embedding","versions":["1"],"platform":"ensemble","inputs":[{"name":"text","datatype":"BYTES","shape":[-1]}],"outputs":[{"name":"output","datatype":"FP32","shape":[-1,384]}]}
```
