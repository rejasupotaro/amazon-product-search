from collections.abc import Callable

import typer
from google.cloud import aiplatform
from kfp import dsl
from kfp.compiler import Compiler
from pipeline.components.dummy import build_evaluate_func, build_preprocess_func, build_train_func
from typing_extensions import Annotated

from amazon_product_search.timestamp import get_unix_timestamp

app = typer.Typer()


@app.command()
def greet():
    print("Hello, World!")


def build_pipeline_func(training_image: str, message: str) -> Callable[[], None]:
    @dsl.pipeline(
        name="dummy",
    )
    def pipeline_func() -> None:
        preprocess_task = build_preprocess_func(image=training_image)(message=message)

        train_task = build_train_func(image=training_image)()
        train_task.after(preprocess_task)

        evaluate_task = build_evaluate_func(image=training_image)()
        evaluate_task.after(train_task)
    return pipeline_func


@app.command()
def run(
    project_id: Annotated[str, typer.Option()],
    region: Annotated[str, typer.Option()],
    service_account: Annotated[str, typer.Option()],
    templates_dir: Annotated[str, typer.Option()],
    training_image: Annotated[str, typer.Option()],
    staging_bucket: Annotated[str, typer.Option()],
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
        pipeline_func=build_pipeline_func(
            training_image=training_image,
            message="Hello World",
        ),
        package_path=package_path,
        # type_check=True,
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

if __name__ == "__main__":
    app()
