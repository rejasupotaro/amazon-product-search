from invoke import task

from amazon_product_search.constants import PROJECT_ID, REGION
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
def index_docs(
    c, index_name, locale="jp", es_host="", extract_keywords=False, encode_text=False, nrows=None, runner="DirectRunner"
):
    command = [
        "poetry run python src/amazon_product_search/indexer/main.py",
        f"--runner={runner}",
        f"--index_name={index_name}",
        f"--locale={locale}",
    ]

    if es_host:
        command.append(f"--es_host={es_host}")

    if extract_keywords:
        command.append("--extract_keywords")

    if encode_text:
        command.append("--encode_text")

    if runner == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            "--direct_num_workers=0",
            "--direct_running_mode=multi_processing",
        ]
    if runner == "DataflowRunner":
        command += [
            f"--project={PROJECT_ID}",
            f"--region={REGION}",
            f"--temp_location=gs://{PROJECT_ID}/temp",
            f"--staging_location=gs://{PROJECT_ID}/staging",
        ]

    if nrows:
        command += [
            f"--nrows={int(nrows)}",
        ]

    c.run(" ".join(command))
