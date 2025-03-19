import contextlib
import subprocess

import typer
from elasticsearch import NotFoundError
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Annotated

from amazon_product_search.constants import HF
from amazon_product_search.es.es_client import EsClient

app = typer.Typer()


def load_config(overrides: list[str]) -> DictConfig:
    with initialize(config_path="../../conf"):
        config= compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(config)
    return config


@app.command()
def delete_index(
    index_name: Annotated[str, typer.Option()],
) -> None:
    es_client = EsClient()
    es_client.delete_index(index_name)
    print(f"{index_name} was deleted.")


@app.command()
def create_index(
    index_name: Annotated[str, typer.Option()],
    locale: Annotated[str, typer.Option()],
) -> None:
    es_client = EsClient()
    with contextlib.suppress(NotFoundError):
        es_client.delete_index(index_name)

    es_client.create_index(locale, index_name)
    print(f"{index_name} was created")


@app.command()
def import_model() -> None:
    es_client = EsClient()
    es_client.import_model(model_id=HF.JA_SBERT)


@app.command()
def transform(
    runner: Annotated[str, typer.Option()],
) -> None:
    """Transform data and load it into the destination (if specified)."""
    overrides = [
        f"runner={runner}",
        "dest=bq",
    ]
    config = load_config(overrides)

    command = [
        "poetry run python src/indexing/doc_pipeline.py",
        f"--locale={config.locale}",
        f"--index_name={config.index_name}",
        f"--runner={config.runner.name}",
        f"--dest={config.dest.name}",
    ]

    if config.encode_text:
        command.append("--encode_text")

    if config.runner.name == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            f"--direct_num_workers={config.runner.num_workers}",
        ]
    elif config.runner.name == "DataflowRunner":
        command += [
            f"--project={config.project_id}",
            f"--region={config.region}",
            f"--num_workers={config.runner.num_workers}",
            f"--worker_machine_type={config.runner.worker_machine_type}",
            "--sdk_location=container",
            f"--sdk_container_image=${config.runner.sdk_container_image}",
            f"--worker_zone={config.runner.worker_zone}",
            f"--temp_location={config.temp_location}",
            f"--staging_location={config.staging_location}",
        ]

    if config.nrows:
        command.append(f"--nrows={config.nrows}")

    subprocess.run(command)


@app.command()
def feed(
    runner: Annotated[str, typer.Option()],
    dest: Annotated[str, typer.Option()],
) -> None:
    """Feed data into the destination."""
    overrides = [
        f"runner={runner}",
        f"dest={dest}",
    ]
    config = load_config(overrides)

    command = [
        "poetry run python src/indexing/feeding_pipeline.py",
        f"--locale={config.locale}",
        f"--index_name={config.index_name}",
        f"--runner={config.runner.name}",
        f"--dest={config.dest.name}",
        f"--dest_host={config.dest.host}",
        f"--table_id={config.table_id}",
    ]

    if config.runner.name == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            f"--direct_num_workers={config.runner.num_workers}",
        ]
    elif config.runner.name == "DataflowRunner":
        config.command += [
            f"--num_workers={config.runner.num_workers}",
            f"--worker_machine_type={config.runner.worker_machine_type}",
            "--sdk_location=container",
            f"--sdk_container_image=${config.runner.sdk_container_image}",
            f"--worker_zone={config.runner.worker_zone}",
        ]

    command += [
        f"--project={config.project_id}",
        f"--region={config.region}",
        f"--temp_location=gs://{config.project_name}/temp",
        f"--staging_location=gs://{config.project_name}/staging",
    ]

    if config.nrows:
        command.append(f"--nrows={config.nrows}")

    subprocess.run(command)


@app.command()
def encode(
    runner: Annotated[str, typer.Option()],
    dest: Annotated[str, typer.Option()],
) -> None:
    """Encode text data and load it into the destination."""
    overrides = [
        f"runner={runner}",
        f"dest={dest}",
    ]
    config = load_config(overrides)

    command = [
        "poetry run python src/indexing/query_pipeline.py",
        f"--locale={config.locale}",
        f"--runner={config.runner.name}",
        f"--dest={config.dest.name}",
        f"--table_id={config.table_id}",
    ]

    if config.runner.name == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            f"--direct_num_workers={config.runner.num_workers}",
        ]
    elif config.runner.name == "DataflowRunner":
        command += [
            f"--num_workers={config.runner.num_workers}",
            f"--worker_machine_type={config.runner.worker_machine_type}",
            "--sdk_location=container",
            f"--sdk_container_image=${config.runner.sdk_container_image}",
            f"--worker_zone={config.runner.worker_zone}",
        ]

    if config.runner.name == "DataflowRunner":
        command += [
            f"--project={config.project_id}",
            f"--region={config.region}",
            f"--temp_location=gs://{config.project_name}/temp",
            f"--staging_location=gs://{config.project_name}/staging",
        ]

    if config.nrows:
        command.append(f"--nrows={config.nrows}")

    subprocess.run(command)
