import json
from textwrap import dedent

from invoke import task

import amazon_product_search.core.vespa.service as vespa_service
from amazon_product_search.constants import HF, VESPA_DIR
from amazon_product_search.core.vespa.vespa_client import VespaClient
from amazon_product_search_dense_retrieval.encoders import SBERTEncoder


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
    res = client.delete_all_docs(content_cluster_name="amazon_content", schema=schema)
    print(res.json)


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
        query_vector = encoder.encode(query_str)
        query_vector = [float(v) for v in query_vector]
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
        json_str = json.dumps(res.json, indent=4, ensure_ascii=False)
        print(json_str)
        print()
