from invoke import task

from amazon_product_search.constants import PROJECT_ID, PROJECT_NAME, REGION
from amazon_product_search.timestamp import get_unix_timestamp


@task
def build_training(c):
    image_uri = f"gcr.io/{PROJECT_ID}/{PROJECT_NAME}/training"
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_DOCKERFILE=Dockerfile.training,_IMAGE={image_uri} \
    """
    c.run(command)


@task
def build_indexing(c):
    image_uri = f"gcr.io/{PROJECT_ID}/{PROJECT_NAME}/pipeline"
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_DOCKERFILE=Dockerfile.indexing,_IMAGE={image_uri} \
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
