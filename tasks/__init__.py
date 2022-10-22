import time

from invoke import Collection, task

from amazon_product_search.constants import IMAGE_URI, PROJECT_ID, REGION
from tasks import data_tasks, es_tasks


def get_unix_timestamp() -> int:
    """Return the current unix timestamp"""
    return int(time.time())


@task
def build_on_cloud(c):
    command = f"""
    gcloud builds submit . \
        --config=cloudbuild.yaml \
        --substitutions=_IMAGE={IMAGE_URI} \
        --timeout=60m
    """
    c.run(command)


@task
def hello_on_cloud(c):
    now = get_unix_timestamp()
    display_name = f"hello-{now}"

    command = f"""
    gcloud ai custom-jobs create \
        --region={REGION} \
        --display-name={display_name} \
        --config=vertexai/hello.yaml
    """
    c.run(command)
    c.run(f"open https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")


@task
def lint(c):
    """Run linters (isort and black, flake8, and mypy)."""
    print("Running isort...")
    c.run("poetry run isort .")

    print("Running black...")
    c.run("poetry run black .")

    print("Running flake8...")
    c.run("poetry run pflake8 src tests tasks")

    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Done")


ns = Collection()
ns.add_task(build_on_cloud)
ns.add_task(hello_on_cloud)
ns.add_task(lint)
ns.add_collection(Collection.from_module(data_tasks, name="data"))
ns.add_collection(Collection.from_module(es_tasks, name="es"))
