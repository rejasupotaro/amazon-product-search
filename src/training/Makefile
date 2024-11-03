REGION?=asia-northeast1
TRAINING_IMAGE_URI:=gcr.io/$(PROJECT_ID)/training:latest
TEMPLATES_DIR:=templates

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
	poetry run python -m training run \
		--project-id $(PROJECT_ID) \
		--region $(REGION) \
		--service-account $(SERVICE_ACCOUNT) \
		--templates-dir $(TEMPLATES_DIR) \
		--staging-bucket $(STAGING_BUCKET)
