
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
    training_image: Annotated[str, typer.Option()],
    pipeline_type: Annotated[str, typer.Option()],
) -> None:
    overrides = [
        f"pipeline_type={pipeline_type}",
        f"compile_parameters.image={training_image}",
        f"runtime_parameters={pipeline_type}",
    ]
    config = load_config(overrides)

    experiment = f"{pipeline_type}-1"
    pipeline_name = f"{pipeline_type}-{get_unix_timestamp()}"

    aiplatform.init(
        project=config.project_id,
        location=config.region,
        staging_bucket=config.staging_bucket,
        experiment=experiment,
    )

    pipeline = PIPELINE_DICT[pipeline_type](config.compile_parameters)
    pipeline.compile(template_path=config.template_filepath)

    job = aiplatform.PipelineJob(
        display_name=pipeline_name,
        template_path=config.template_filepath,
        parameter_values=config.runtime_parameters,
    )
    job.submit(
        service_account=config.service_account,
        experiment=experiment,
    )
    job._block_until_complete()

if __name__ == "__main__":
    app()
