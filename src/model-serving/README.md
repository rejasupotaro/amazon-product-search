# Triton Inference Server

This directory contains a text embedding model and its server.

## Installation

This package is part of the Amazon Product Search workspace. Install from the root directory:

```shell
# From the root directory
$ uv sync
```

## Export Model

Before launching the server, you need to convert the text embedding model to ONNX format. Run the following command:

```
make export
```

Then, move the exported ONNX model to `model_repository/embedder/1/model.onnx`.

## Launch Server

To start the Triton Inference Server, run these commands:

```
docker compose build
docker compose up
```

## Send Request

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

## Benchmark

You can benchmark the Triton server by running `make benchmark`:

```
$ make benchmark
Summary:
  Count:	100
  Total:	1.89 s
  Slowest:	75.24 ms
  Fastest:	34.75 ms
  Average:	37.29 ms
  Requests/sec:	52.90

Response time histogram:
  34.751 [1]  |
  38.800 [93] |∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
  42.850 [3]  |∎
  46.899 [1]  |
  50.948 [1]  |
  54.998 [0]  |
  59.047 [0]  |
  63.097 [0]  |
  67.146 [0]  |
  71.195 [0]  |
  75.245 [1]  |

Latency distribution:
  10 % in 35.70 ms
  25 % in 35.99 ms
  50 % in 36.31 ms
  75 % in 37.07 ms
  90 % in 37.93 ms
  95 % in 40.71 ms
  99 % in 49.94 ms

Status code distribution:
  [OK]   100 responses
```

## Debugging Tips

You can adjust the logging verbosity by calling the logging API:

```
$ curl -X POST localhost:8000/v2/logging \
     -d '{"log_verbose_level":1}' -H "Content-Type: application/json"
```

Setting `log_verbose_level` to 0 disables verbose logging in the Triton server.

## Host on Vertex AI

[Serving Predictions with NVIDIA Triton  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/predictions/using-nvidia-triton)
