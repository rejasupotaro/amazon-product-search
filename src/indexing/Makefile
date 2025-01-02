INDEXING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/indexing:latest

.PHONY: build
build_indexing:
	gcloud builds submit . \
		--config=cloudbuild.yaml \
		--substitutions=_IMAGE=${INDEXING_IMAGE_URI}