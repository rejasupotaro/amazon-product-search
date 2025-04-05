# -------------------------------------
#  Execution Commands
# -------------------------------------
.PHONY: lint
lint:
	poetry run ruff check --fix --unsafe-fixes --show-fixes
	poetry run mypy src/amazon-product-search/src --explicit-package-bases --namespace-packages
	poetry run mypy src/data-source/src --explicit-package-bases --namespace-packages
	poetry run mypy src/dense-retrieval/src --explicit-package-bases --namespace-packages
	poetry run mypy src/indexing/src --explicit-package-bases --namespace-packages
	poetry run mypy src/model-serving/src --explicit-package-bases --namespace-packages
	poetry run mypy src/training/src --explicit-package-bases --namespace-packages
