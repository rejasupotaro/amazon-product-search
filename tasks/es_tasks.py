import contextlib

from invoke import task

from amazon_product_search.constants import HF
from amazon_product_search.core.es.es_client import EsClient
from elasticsearch import NotFoundError


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
    with contextlib.suppress(NotFoundError):
        es_client.delete_index(index_name=index_name)

    es_client.create_index(index_name=index_name)


@task
def import_model(c):
    es_client = EsClient()
    es_client.import_model(model_id=HF.JA_SBERT)
