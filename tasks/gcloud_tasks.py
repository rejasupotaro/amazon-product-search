from invoke import task

from amazon_product_search.constants import PROJECT_ID, PROJECT_NAME, REGION
from amazon_product_search.timestamp import get_unix_timestamp


@task
def build_training(c):
    image_uri = f"gcr.io/{PROJECT_ID}/{PROJECT_NAME}/training"
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_IMAGE={image_uri} \
        --timeout=60m
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
        --config=vertex_ai/hello.yaml
    """
    c.run(command)
    c.run(f"open https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
