from textwrap import dedent

from invoke import task

import amazon_product_search.vespa.service as vespa_service
from amazon_product_search.constants import HF, VESPA_DIR
from amazon_product_search.vespa.vespa_client import VespaClient
from dense_retrieval.encoders import SBERTEncoder

"""
To run Vespa locally, execute the following commands:
```
$ docker compose --profile vespa up
$ vespa deploy ./vespa
$ poetry run inv indexing.feed \
    --locale=us \
    --dest=vespa \
    --dest-host=http://localhost:8080 \
    --index-name=product
```
"""


@task
def start(c):
    vespa_service.start()


@task
def stop(c):
    vespa_service.stop()


@task
def restart(c):
    vespa_service.stop()
    vespa_app = vespa_service.start()
    vespa_app.application_package.to_files(VESPA_DIR)


@task
def delete_all_docs(c, schema):
    client = VespaClient()
    response = client.delete_all_docs(content_cluster_name="amazon_content", schema=schema)
    print(response)


@task
def search(c):
    # https://docs.vespa.ai/en/nearest-neighbor-search-guide.html
    # https://docs.vespa.ai/en/embedding.html
    # https://docs.vespa.ai/en/phased-ranking.html
    client = VespaClient()
    encoder = SBERTEncoder(HF.JP_SLUKE_MEAN)
    while True:
        query_str = input("query: ")
        if not query_str:
            break
        query_vector = encoder.encode(query_str).tolist()
        yql = dedent(
            """
            select *
            from product
            where userQuery() or ({targetHits:1}nearestNeighbor(product_vector,query_vector))
            """
        ).strip()
        query = {
            "yql": yql,
            "query": query_str,
            "input.query(query_vector)": query_vector,
            "ranking.profile": "hybrid",
            "hits": 10,
        }
        res = client.search(query)
        print(res)
        print()
