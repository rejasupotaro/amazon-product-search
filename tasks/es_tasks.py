from invoke import task

from amazon_product_search.constants import HF, PROJECT_ID, PROJECT_NAME, REGION
from amazon_product_search.es.es_client import EsClient


@task
def delete_index(c, index_name):
    es_client = EsClient()
    es_client.delete_index(index_name=index_name)
    print(f"{index_name} was deleted.")


@task
def create_index(c, index_name):
    es_client = EsClient()
    es_client.create_index(index_name=index_name)
    print(f"{index_name} was created.")


@task
def recreate_index(c, index_name):
    es_client = EsClient()
    es_client.delete_index(index_name=index_name)
    es_client.create_index(index_name=index_name)


@task
def import_model(c):
    es_client = EsClient()
    es_client.import_model(model_id=HF.JA_SBERT)


@task
def index(
    c,
    index_name,
    locale="jp",
    dest_host="",
    extract_keywords=False,
    encode_text=False,
    nrows=None,
    runner="DirectRunner",
):
    command = [
        "poetry run python src/amazon_product_search/indexer/main.py",
        f"--runner={runner}",
        f"--index_name={index_name}",
        f"--locale={locale}",
    ]

    if dest_host:
        command.append("--dest=es")
        command.append(f"--dest_host={dest_host}")

    if extract_keywords:
        command.append("--extract_keywords")

    if encode_text:
        command.append("--encode_text")

    if runner == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            "--direct_num_workers=0",
        ]
    elif runner == "DataflowRunner":
        command += [
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--temp_location=gs://{PROJECT_NAME}/temp",
            f"--staging_location=gs://{PROJECT_NAME}/staging",
        ]

    if nrows:
        command.append(f"--nrows={int(nrows)}")

    c.run(" ".join(command))
