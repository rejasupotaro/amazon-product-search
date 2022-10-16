import time

from invoke import task

import misc
from amazon_product_search.constants import IMAGE_URI, PROJECT_ID, REGION


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
def index(c, runner="DirectRunner", nrows=None):
    command = [
        "poetry run python src/indexer/main.py",
        f"--runner={runner}",
        "--locale=jp",
        "--es_host=http://localhost:9200",
    ]

    if runner == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            "--direct_num_workers=0",
            "--direct_running_mode=multi_processing",
        ]

    if nrows:
        command += [
            f"--nrows={nrows}",
        ]

    c.run(" ".join(command))


@task
def lint(c):
    """Run linters (isort and black, flake8, and mypy)."""
    print("Running isort...")
    c.run("poetry run isort .")

    print("Running black...")
    c.run("poetry run black .")

    print("Running flake8...")
    c.run("poetry run pflake8 src tests tasks.py")

    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Done")


@task
def split_dataset_by_locale(c):
    misc.split_dataset_by_locale()
