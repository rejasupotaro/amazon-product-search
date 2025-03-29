# Triton Inference Server

This directory contains a text embedding model and its server.

## Export Model

Before launching the server, convert a text embedding model into the ONNX format. To do this, execute the following command:

```
make export
```

Then, move the exported ONNX model to `model_repository/embedder/1/model.onnx`.

## Launch Server

To launch the Triton Inference Server, execute the following commands:

```
docker compose build
docker compose up
```

It launches the following services:

```
$ curl http://localhost:8000/v2/models/tokenizer
{"name":"tokenizer","versions":["1"],"platform":"python","inputs":[{"name":"text","datatype":"BYTES","shape":[-1]}],"outputs":[{"name":"input_ids","datatype":"INT64","shape":[-1,-1]},{"name":"attention_mask","datatype":"INT64","shape":[-1,-1]}]}
$ curl -X POST http://localhost:8000/v2/models/tokenizer/infer \
-H "Content-Type: application/json" \
-d '{
  "inputs": [
    {
      "name": "text",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["Hello World"]
    }
  ]
}'
{"model_name":"tokenizer","model_version":"1","outputs":[{"name":"input_ids","datatype":"INT64","shape":[1,6],"data":[2,1471,4424,10273,5931,3]},{"name":"attention_mask","datatype":"INT64","shape":[1,6],"data":[1,1,1,1,1,1]}]}
```

```
$ curl http://localhost:8000/v2/models/embedder
{"name":"embedder","versions":["1"],"platform":"onnxruntime_onnx","inputs":[{"name":"input_ids","datatype":"INT64","shape":[-1,-1]},{"name":"attention_mask","datatype":"INT64","shape":[-1,-1]}],"outputs":[{"name":"output","datatype":"FP32","shape":[-1,-1]}]}
$ curl -X POST http://localhost:8000/v2/models/embedder/infer \
-H "Content-Type: application/json" \
-d '{
  "inputs": [
    {
      "name": "input_ids",
      "datatype": "INT64",
      "shape": [1, 6],
      "data": [2,1471,4424,10273,5931,3]
    },
    {
      "name": "attention_mask",
      "datatype": "INT64",
      "shape": [1, 6],
      "data": [1,1,1,1,1,1]
    }
  ]
}'
```

```
$ curl http://localhost:8000/v2/models/ensemble
{"name":"ensemble","versions":["1"],"platform":"ensemble","inputs":[{"name":"text","datatype":"BYTES","shape":[-1]}],"outputs":[{"name":"output","datatype":"FP32","shape":[-1,-1]}]}
$ curl -X POST http://localhost:8000/v2/models/ensemble/infer \
-H "Content-Type: application/json" \
-d '{
  "inputs": [
    {
      "name": "text",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["Hello World"]
    }
  ]
}'
```

## Host on Vertex AI

[Serving Predictions with NVIDIA Triton  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/predictions/using-nvidia-triton)
