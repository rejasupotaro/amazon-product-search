export

INDEX_NAME?=products_jp
LOCALE?=jp
INDEXING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/indexing:latest

RUNNER?=direct
DEST?=bq

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

.PHONY: transform
transform:
	poetry run python -m indexing transform \
		--runner=$(RUNNER)

.PHONY: feed
feed:
	poetry run python -m indexing feed \
		--runner=$(RUNNER)
		--dest=$(DEST)

.PHONY: encode
encode:
	poetry run python -m indexing encode \
		--runner=$(RUNNER)
		--dest=$(DEST)