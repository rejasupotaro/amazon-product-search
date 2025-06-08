# -------------------------------------
#  Execution Commands
# -------------------------------------
.PHONY: lint
lint:
	uv run ruff check --fix --unsafe-fixes --show-fixes
	uv run mypy src/amazon-product-search/src --explicit-package-bases --namespace-packages
	uv run mypy src/data-source/src --explicit-package-bases --namespace-packages
	uv run mypy src/dense-retrieval/src --explicit-package-bases --namespace-packages
	uv run mypy src/indexing/src --explicit-package-bases --namespace-packages
	uv run mypy src/model-serving/src --explicit-package-bases --namespace-packages
	uv run mypy src/training/src --explicit-package-bases --namespace-packages
