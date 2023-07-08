from invoke import task

from amazon_product_search.constants import (
    INDEXING_IMAGE_URI,
    REGION,
    TRAINING_IMAGE_URI,
)
from amazon_product_search.core.timestamp import get_unix_timestamp


@task
def build_training(c):
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_DOCKERFILE=Dockerfile.training,_IMAGE={TRAINING_IMAGE_URI} \
    """
    c.run(command)


@task
def build_indexing(c):
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_DOCKERFILE=Dockerfile.indexing,_IMAGE={INDEXING_IMAGE_URI} \
    """
    c.run(command)


@task
def train_encoder(c):
    now = get_unix_timestamp()
    display_name = f"train-encoder-{now}"
    command = f"""
    gcloud ai custom-jobs create \
        --region={REGION} \
        --display-name={display_name} \
        --config=vertex/train_encoder.yaml
    """
    c.run(command)


@task
def hello(c):
    now = get_unix_timestamp()
    display_name = f"hello-{now}"
    command = f"""
    gcloud ai custom-jobs create \
        --region={REGION} \
        --display-name={display_name} \
        --config=vertex/hello.yaml
    """
    c.run(command)
