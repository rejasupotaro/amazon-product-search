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


def load_config(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize(config_path="../../conf"):
        config= compose(config_name=config_name, overrides=overrides)
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
def import_model(
) -> None:
    es_client = EsClient()
    es_client.import_model(model_id=HF.JA_SBERT)


@app.command()
def transform(
) -> None:
    config = load_config("transform", [])

    command = [
        "poetry run python src/indexing/doc_pipeline.py",
        f"--locale={config.locale}",
        f"--index_name={config.index_name}",
        f"--runner={config.runner}",
        f"--dest={config.dest}",
        f"--dest_host={config.dest_host}",
    ]

    if config.encode_text:
        command.append("--encode_text")

    if config.runner == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            "--direct_num_workers=0",
        ]
    elif config.runner == "DataflowRunner":
        command += [
            "--num_workers=1",
            "--worker_machine_type=n2-highmem-8",
            "--sdk_location=container",
            f"--sdk_container_image=gcr.io/{config.project_id}/{config.project_name}/indexing",
            f"--worker_zone={config.region}-c",
        ]

    if (config.runner == "DataflowRunner") or (config.dest == "bq"):
        command += [
            f"--project={config.project_id}",
            f"--region={config.region}",
            f"--temp_location=gs://{config.project_name}/temp",
            f"--staging_location=gs://{config.project_name}/staging",
        ]

    if config.dest == "bq" and config.table_id:
        command += [
            f"--table_id={config.table_id}",
        ]

    if config.nrows:
        command.append(f"--nrows={config.nrows}")

    subprocess.run(command)
