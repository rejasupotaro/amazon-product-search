from invoke import task

import amazon_product_search.training.dummy.pipeline as dummy_pipeline
import amazon_product_search.training.fine_tuning_cl.pipeline as fine_tuning_cl_pipeline
import amazon_product_search.training.fine_tuning_mlm.pipeline as fine_tuning_mlm_pipeline
from amazon_product_search.constants import (
    TRAINING_IMAGE_URI,
)


@task
def build(c):
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_DOCKERFILE=Dockerfile.training,_IMAGE={TRAINING_IMAGE_URI} \
    """
    c.run(command)


@task
def dummy(c):
    dummy_pipeline.main()


@task
def fine_tune_cl(c):
    fine_tuning_cl_pipeline.main()


@task
def fine_tune_mlm(c):
    fine_tuning_mlm_pipeline.main()
