# -------------------------------------
#  Execution Commands
# -------------------------------------
.PHONY: lint
lint:
	python -m ruff check --fix --unsafe-fixes --show-fixes
