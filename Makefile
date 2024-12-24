REGION?=asia-northeast1
TRAINING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/training:latest
INDEXING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/indexing:latest
TEMPLATES_DIR:=templates

# -------------------------------------
#  Execution Commands
# -------------------------------------
.PHONY: lint
lint:
	poetry run ruff check --fix --unsafe-fixes --show-fixes
	poetry run mypy src/dense-retrieval/src --explicit-package-bases --namespace-packages
	poetry run mypy src/amazon-product-search/src --explicit-package-bases --namespace-packages
	poetry run mypy src/training/src --explicit-package-bases --namespace-packages
	poetry run mypy src/indexing/src --explicit-package-bases --namespace-packages

.PHONY: build_training
build_training:
	gcloud builds submit . \
		--config=cloudbuild.yaml \
		--substitutions=_DOCKERFILE=src/training/Dockerfile,_IMAGE=${TRAINING_IMAGE_URI}

.PHONY: build_indexing
build_indexing:
	gcloud builds submit . \
		--config=cloudbuild.yaml \
		--substitutions=_DOCKERFILE=src/indexing/Dockerfile,_IMAGE=${INDEXING_IMAGE_URI}
