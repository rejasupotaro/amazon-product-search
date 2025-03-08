REGION?=asia-northeast1
TRAINING_IMAGE:=gcr.io/$(PROJECT_ID)/training:latest
TEMPLATES_DIR:=templates
PIPELINE_TYPE?=dummy

# -------------------------------------
#  Execution Commands
# -------------------------------------
.PHONY: build
build:
	gcloud builds submit . \
		--config=cloudbuild.yaml \
		--substitutions=_IMAGE=${TRAINING_IMAGE_URI}

.PHONY: run
run:
	poetry run python -m pipeline run \
		--project-id=$(PROJECT_ID) \
		--project-name=$(PROJECT_NAME) \
		--region=$(REGION) \
		--service-account=$(SERVICE_ACCOUNT) \
		--templates-dir=$(TEMPLATES_DIR) \
		--training-image=$(TRAINING_IMAGE) \
		--staging-bucket=$(STAGING_BUCKET) \
		--pipeline-type=$(PIPELINE_TYPE)
