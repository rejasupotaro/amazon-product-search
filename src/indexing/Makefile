INDEX_NAME?=products_jp
LOCALE?=jp
INDEXING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/indexing:latest

.PHONY: build
build:
	gcloud builds submit . \
		--config=cloudbuild.yaml \
		--substitutions=_IMAGE=$(INDEXING_IMAGE_URI)

.PHONY: create-index
create-index:
	poetry run python -m indexing create-index \
		--index-name=$(INDEX_NAME) \
		--locale=$(LOCALE)