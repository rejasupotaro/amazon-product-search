from google.cloud import aiplatform
from kfp import dsl
from kfp.compiler import Compiler

from amazon_product_search.timestamp import get_unix_timestamp
from training.dummy.components.evaluate import build_evaluate_func
from training.dummy.components.predict import build_predict_func
from training.dummy.components.preprocess import build_preprocess_func
from training.dummy.components.train import build_train_func


@dsl.pipeline(
    name="dummy",
)
def pipeline_func(training_image: str, message: str) -> None:
    preprocess_task = build_preprocess_func(image=training_image)(message=message)

    train_task = build_train_func(image=training_image)()
    train_task.after(preprocess_task)

    evaluate_task = build_evaluate_func()()
    evaluate_task.after(train_task)

    predict_task = build_predict_func()()
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
