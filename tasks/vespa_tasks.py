from invoke import task

import amazon_product_search.vespa.service as vespa_service
from amazon_product_search.constants import PROJECT_ID, PROJECT_NAME, REGION
from amazon_product_search.vespa.vespa_client import VespaClient


@task
def start(c):
    vespa_service.start()


@task
def stop(c):
    vespa_service.stop()


@task
def delete_all_docs(c, schema):
    client = VespaClient()
    res = client.delete_all_docs(content_cluster_name="amazon", schema=schema)
    print(res.json)
    print(res.raw)
    print(res.reason)


@task
def search(c, query):
    client = VespaClient()
    res = client.search(query)
    print(res.json)


@task
def index(
    c,
    schema,
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
        f"--index_name={schema}",
        f"--locale={locale}",
    ]

    if dest_host:
        command.append("--dest=vespa")
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
