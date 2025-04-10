import contextlib

import typer
from elasticsearch import NotFoundError
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from typing_extensions import Annotated

from amazon_product_search.constants import HF
from amazon_product_search.es.es_client import EsClient
from indexing.options import IndexerOptions
from indexing.pipelines.pipelien_types import PipelineType

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
def query(
    pipeline_type: Annotated[PipelineType, typer.Option()],
    runner: Annotated[str, typer.Option()],
    dest: Annotated[str, typer.Option()],
) -> None:
    """Encode text data and load it into the destination."""
    overrides = [
        f"runner={runner}",
        f"dest={dest}",
    ]
    config = load_config(overrides)

    kwargs = {
        "pipeline_type": str(pipeline_type),
        "locale": config.locale,
        "runner": config.runner.name,
        "dest": config.dest.name,
        "table_id": config.table_id,
        "data_dir": config.data_dir,
    }

    if config.runner.name == "DirectRunner":
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
        kwargs["direct_num_workers"] = config.runner.num_workers
    elif config.runner.name == "DataflowRunner":
        kwargs |= {
            "num_workers": config.runner.num_workers,
            "worker_machine_type": config.runner.worker_machine_type,
            "sdk_location": "container",
            "sdk_container_image": config.runner.sdk_container_image,
            "worker_zone": config.runner.worker_zone,
        }

    if config.runner.name == "DataflowRunner":
        kwargs |= {
            "project": config.project_id,
            "region": config.region,
            "temp_location": f"gs://{config.project_name}/temp",
            "staging_location": f"gs://{config.project_name}/staging",
        }

    if config.nrows:
        kwargs["nrows"] = config.nrows

    options = IndexerOptions(**kwargs)
    pipeline = PipelineType(options.pipeline_type).pipeline_class(options)
    pipeline.run()


@app.command()
def doc(
    pipeline_type: Annotated[PipelineType, typer.Option()],
    runner: Annotated[str, typer.Option()],
    dest: Annotated[str, typer.Option()],
) -> None:
    """Transform data and load it into the destination (if specified)."""
    overrides = [
        f"runner={runner}",
        f"dest={dest}",
    ]
    config = load_config(overrides)

    kwargs = {
        "pipeline_type": str(pipeline_type),
        "locale": config.locale,
        "index_name": config.index_name,
        "runner": config.runner.name,
        "dest": config.dest.name,
        "data_dir": config.data_dir,
    }

    if config.encode_text:
        kwargs["encode_text"] = True

    if config.runner.name == "DirectRunner":
        # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
        kwargs["direct_num_workers"] = config.runner.num_workers
    elif config.runner.name == "DataflowRunner":
        kwargs |= {
            "project": config.project_id,
            "region": config.region,
            "num_workers": config.runner.num_workers,
            "worker_machine_type": config.runner.worker_machine_type,
            "sdk_location": "container",
            "sdk_container_image": config.runner.sdk_container_image,
            "worker_zone": config.runner.worker_zone,
            "temp_location": config.temp_location,
            "staging_location": config.staging_location,
        }

    if config.nrows:
        kwargs["nrows"] = config.nrows

    options = IndexerOptions(**kwargs)
    pipeline = PipelineType(options.pipeline_type).pipeline_class(options)
    pipeline.run()


@app.command()
def feed(
    pipeline_type: Annotated[PipelineType, typer.Option()],
    runner: Annotated[str, typer.Option()],
    dest: Annotated[str, typer.Option()],
) -> None:
    """Feed data into the destination."""
    overrides = [
        f"runner={runner}",
        f"dest={dest}",
    ]
    config = load_config(overrides)

    kwargs = {
        "pipeline_type": str(pipeline_type),
        "locale": config.locale,
        "index_name": config.index_name,
        "runner": config.runner.name,
        "dest": config.dest.name,
        "table_id": config.table_id,
    }

    if config.dest.name != "stdout":
        kwargs["dest_host"] = config.dest.host

    if config.runner.name == "DirectRunner":
        # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
        kwargs["direct_num_workers"] = config.runner.num_workers
    elif config.runner.name == "DataflowRunner":
        kwargs |= {
            "num_workers": config.runner.num_workers,
            "worker_machine_type": config.runner.worker_machine_type,
            "sdk_location": "container",
            "sdk_container_image": config.runner.sdk_container_image,
            "worker_zone": config.runner.worker_zone,
        }

    kwargs |= {
        "project": config.project_id,
        "region": config.region,
        "temp_location": f"gs://{config.project_name}/temp",
        "staging_location": f"gs://{config.project_name}/staging",
    }

    if config.nrows:
        kwargs["nrows"] = config.nrows

    options = IndexerOptions(**kwargs)
    pipeline = PipelineType(options.pipeline_type).pipeline_class(options)
    pipeline.run()
