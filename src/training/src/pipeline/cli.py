
import typer
from google.cloud import aiplatform
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pipeline.pipelines import PIPELINE_DICT
from typing_extensions import Annotated

from amazon_product_search.timestamp import get_unix_timestamp

app = typer.Typer()


def load_config(overrides: list[str]) -> DictConfig:
    with initialize(config_path="../../conf"):
        config= compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(config)
    return config

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
    experiment = f"{pipeline_type}-1"
    pipeline_name = f"{pipeline_type}-{get_unix_timestamp()}"
    template_path = f"{templates_dir}/{pipeline_type}.yaml"

    overrides = [
        f"project_id={project_id}",
        f"project_dir=gs://{project_name}",
        f"pipeline_type={pipeline_type}",
        f"compile_parameters.image={training_image}",
        f"runtime_parameters={pipeline_type}",
    ]

    config = load_config(overrides)

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket=staging_bucket,
        experiment=experiment,
    )

    pipeline = PIPELINE_DICT[pipeline_type](config.compile_parameters)
    pipeline.compile(template_path=template_path)

    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=template_path,
        parameter_values=config.runtime_parameters,
    )
    job.submit(
        service_account=service_account,
        experiment=experiment,
    )
    job._block_until_complete()

if __name__ == "__main__":
    app()
