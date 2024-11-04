from google.cloud import aiplatform
from kfp import dsl
from kfp.compiler import Compiler
from kfp.dsl import Metrics, Output

from amazon_product_search.constants import (
    TRAINING_IMAGE_URI,
)
from amazon_product_search.timestamp import get_unix_timestamp


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
def evaluate(metrics_output: Output[Metrics]) -> None:
    print("Evaluate")
    metrics_output.log_metric("loss", [0.2, 0.1])
    metrics_output.log_metric("acc", [0.8, 0.9])


@dsl.component
def predict() -> None:
    print("Predict")


@dsl.pipeline(
    name="dummy",
)
def pipeline_func(training_image: str, message: str) -> None:
    preprocess(message=message)

    train_task = train()
    train_task.after(preprocess)

    evaluate_task = evaluate()
    evaluate_task.after(train_task)

    predict_task = predict()
    predict_task.after(evaluate_task)


def run(
    project_id: str,
    region: str,
    service_account: str,
    templates_dir: str,
    training_image: str,
    staging_bucket: str,
) -> None:
    """Invoke a Vertex AI custom training job.

    Run `poetry run inv gcloud.build-training` to build the image used for training in advance.

    To create a job, run `poetry run python src/amazon_product_search/training/dummy/pipeline.py`.
    It first compiles the pipeline into the YAML format.
    The YAML file includes all information for executing the pipeline on Vertex AI pipelines.

    For more details:
    * https://cloud.google.com/vertex-ai/docs/pipelines/build-pipeline
    """
    experiment = "dummy-1"
    display_name = f"dummy-{get_unix_timestamp()}"
    package_path = f"{templates_dir}/dummy.yaml"

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
        experiment=experiment,
    )

    Compiler().compile(
        pipeline_func=pipeline_func,
        package_path=package_path,
        type_check=True,
        pipeline_parameters={
            "training_image": training_image,
            "message": "Hello World",
        },
    )

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=package_path,
    )
    job.submit(
        service_account=service_account,
        experiment=experiment,
    )
    job._block_until_complete()
