from google.cloud import aiplatform
from kfp import dsl
from kfp.compiler import Compiler

from amazon_product_search.constants import (
    PROJECT_ID,
    REGION,
    SERVICE_ACCOUNT,
    STAGING_BUCKET,
    TRAINING_IMAGE_URI,
    VERTEX_DIR,
)
from amazon_product_search.core.timestamp import get_unix_timestamp


@dsl.component(
    base_image=TRAINING_IMAGE_URI,
    install_kfp_package=False,
)
def preprocess(message: str) -> None:
    import logging

    logging.info(message)


@dsl.container_component
def train() -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python"],
        args=["-c", "print('Hello World')"],
    )


@dsl.component
def evaluate() -> None:
    print("Evaluate")


@dsl.component
def predict() -> None:
    print("Predict")


@dsl.pipeline(
    name="example",
)
def pipeline_func(message: str) -> None:
    preprocess(message=message)

    train_task = train()
    train_task.after(preprocess)

    evaluate_task = evaluate()
    evaluate_task.after(train_task)

    predict_task = predict()
    predict_task.after(evaluate_task)


def main() -> None:
    """Invoke a Vertex AI custom training job.

    Run `poetry run inv gcloud.build-training` to build the image used for training in advance.

    To create a job, run `poetry run python src/amazon_product_search/training/example/pipeline.py`.
    It first compiles the pipeline into the YAML format.
    The YAML file includes all information for executing the pipeline on Vertex AI pipelines.

    For more details:
    * https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline
    """
    experiment = f"example-{get_unix_timestamp()}"
    package_path = f"{VERTEX_DIR}/example.yaml"

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
        experiment=experiment,
    )

    Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=package_path,
        type_check=True,
        pipeline_parameters={
            "message": "Hello World",
        },
    )

    job = aiplatform.PipelineJob(
        display_name=experiment,
        template_path=package_path,
    )
    job.submit(
        service_account=SERVICE_ACCOUNT,
        experiment=experiment,
    )
    job._block_until_complete()


if __name__ == "__main__":
    main()
