REGION?=asia-northeast1
TRAINING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/training:latest
TEMPLATES_DIR:=templates

# -------------------------------------
#  Execution Commands
# -------------------------------------
.PHONY: lint
lint:
	python -m ruff check --fix --unsafe-fixes --show-fixes
	python -m mypy src/dense-retrieval/src --explicit-package-bases --namespace-packages
	python -m mypy src/amazon-product-search/src --explicit-package-bases --namespace-packages
	python -m mypy src/training/src --explicit-package-bases --namespace-packages

.PHONY: build
build:
	gcloud builds submit . \
		--config=cloudbuild.yaml \
		--substitutions=_DOCKERFILE=src/training/Dockerfile,_IMAGE=${TRAINING_IMAGE_URI}
