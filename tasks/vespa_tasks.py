from invoke import task

import amazon_product_search.vespa.service as vespa_service
from amazon_product_search.vespa.vespa_client import VespaClient


@task
def start(c):
    vespa_service.start()


@task
def stop(c):
    vespa_service.stop()


@task
def delete_docs(c):
    client = VespaClient()
    res = client.delete_all_docs(content_cluster_name="amazon", schema="product")
    print(res)


@task
def index(c):
    client = VespaClient()
    products = [
        {
            "product_id": "1",
            "product_title": "product title",
            "product_description": "product description",
            "product_brand": "product brand",
        }
    ]
    batch = [{"id": product["product_id"], "fields": product} for product in products]
    res = client.feed(schema="product", batch=batch)
    print(res.json)


@task
def search(c):
    client = VespaClient()
    res = client.search("title")
    print(res.json)
