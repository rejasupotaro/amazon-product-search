export

INDEX_NAME?=products_jp
LOCALE?=jp
INDEXING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/indexing:latest

RUNNER?=direct
DEST?=stdout

.PHONY: build
build:
	gcloud builds submit . \
		--config=cloudbuild.yaml \
		--substitutions=_IMAGE=$(INDEXING_IMAGE_URI)

.PHONY: delete-index
delete-index:
	poetry run python -m indexing delete-index \
		--index-name=$(INDEX_NAME) \

.PHONY: create-index
create-index:
	poetry run python -m indexing create-index \
		--index-name=$(INDEX_NAME) \
		--locale=$(LOCALE)

.PHONY: import-model
import-model:
	poetry run python -m indexing import-model

.PHONY: query-pipeline
query-pipeline:
	poetry run python -m indexing query \
		--pipeline-type=query \
		--runner=$(RUNNER) \
		--dest=$(DEST)

.PHONY: doc-pipeline
doc-pipeline:
	poetry run python -m indexing doc \
		--pipeline-type=doc \
		--runner=$(RUNNER) \
		--dest=$(DEST)

.PHONY: feed-pipeline
feed-pipeline:
	poetry run python -m indexing feed \
		--pipeline-type=feed \
		--runner=$(RUNNER) \
		--dest=$(DEST)