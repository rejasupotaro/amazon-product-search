# -------------------------------------
#  Execution Commands
# -------------------------------------
.PHONY: lint
lint:
	python -m ruff check --fix --unsafe-fixes --show-fixes
	python -m mypy src/amazon-product-search/src --explicit-package-bases  --namespace-packages
