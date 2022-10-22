from invoke import task

from amazon_product_search.es.es_client import EsClient


@task
def delete_index(c, locale="jp"):
    es_client = EsClient()
    index_name = f"products_{locale}"
    es_client.delete_index(index_name=index_name)


@task
def create_index(c, locale="jp"):
    es_client = EsClient()
    index_name = f"products_{locale}"
    es_client.create_index(index_name=index_name)


@task
def recreate_index(c, locale="jp"):
    es_client = EsClient()
    index_name = f"products_{locale}"
    es_client.delete_index(index_name=index_name)
    es_client.create_index(index_name=index_name)


@task
def index_docs(c, runner="DirectRunner", locale="jp", nrows=None):
    command = [
        "poetry run python src/amazon_product_search/sparse_retrieval/indexer/main.py",
        f"--runner={runner}",
        f"--locale={locale}",
        "--es_host=http://localhost:9200",
    ]

    if runner == "DirectRunner":
        command += [
            # https://github.com/apache/beam/blob/master/sdks/python/apache_beam/options/pipeline_options.py#L617-L621
            "--direct_num_workers=0",
            "--direct_running_mode=multi_processing",
        ]

    if nrows:
        command += [
            f"--nrows={int(nrows)}",
        ]

    c.run(" ".join(command))
