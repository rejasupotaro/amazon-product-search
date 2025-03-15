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
		--substitutions=_IMAGE=${TRAINING_IMAGE}

.PHONY: run
run:
	poetry run python -m pipeline run \
		--training-image=$(TRAINING_IMAGE) \
		--pipeline-type=$(PIPELINE_TYPE)
