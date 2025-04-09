.PHONY: export
export:
	poetry run python src/model_serving/cli.py

.PHONY: benchmark
benchmark:
	@if [ ! -f /tmp/open_inference_grpc.proto ]; then \
		echo "Downloading proto file to /tmp directory..."; \
		curl -sSf -o /tmp/open_inference_grpc.proto https://raw.githubusercontent.com/kserve/open-inference-protocol/main/specification/protocol/open_inference_grpc.proto; \
	fi
	@ghz \
		--proto=/tmp/open_inference_grpc.proto \
		--call=inference.GRPCInferenceService.ModelInfer \
		--data '{"model_name": "ensemble", "inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "contents": {"bytes_contents": ["aGVsbG8gd29ybGQ="]}}]}' \
		--total=100 \
		--concurrency=2 \
		--insecure \
		localhost:8001
