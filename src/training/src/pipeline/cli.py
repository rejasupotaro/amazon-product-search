
import typer
from google.cloud import aiplatform
from pipeline.pipelines import PIPELINE_DICT
from typing_extensions import Annotated

from amazon_product_search.timestamp import get_unix_timestamp

app = typer.Typer()


@app.command()
def greet():
    print("Hello, World!")


@app.command()
def run(
    project_id: Annotated[str, typer.Option()],
    project_name: Annotated[str, typer.Option()],
    region: Annotated[str, typer.Option()],
    service_account: Annotated[str, typer.Option()],
    templates_dir: Annotated[str, typer.Option()],
    training_image: Annotated[str, typer.Option()],
    staging_bucket: Annotated[str, typer.Option()],
    pipeline_type: Annotated[str, typer.Option()],
) -> None:
    experiment = "dummy-1"
    pipeline_name = f"dummy-{get_unix_timestamp()}"
    template_path = f"{templates_dir}/{pipeline_type}.yaml"

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
        experiment=experiment,
    )

    pipeline = PIPELINE_DICT[pipeline_type](image=training_image)
    pipeline.compile(template_path=template_path)

    project_dir = f"gs://{project_name}"
    runtime_parameters = {
        "project_dir": project_dir,
        "input_filename": "merged_us.parquet",
        "bert_model_name": "cl-tohoku/bert-base-japanese-char-v2",
    }

    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=template_path,
        parameter_values=runtime_parameters,
    )
    job.submit(
        service_account=service_account,
        experiment=experiment,
    )
    job._block_until_complete()

if __name__ == "__main__":
    app()
