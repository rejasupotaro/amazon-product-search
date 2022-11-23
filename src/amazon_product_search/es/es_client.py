import json
from typing import Any, Callable, Iterator, Optional

from elasticsearch import Elasticsearch, helpers

from amazon_product_search.es.response import Response, Result


class EsClient:
    """A wrapper class of https://elasticsearch-py.readthedocs.io/"""

    def __init__(self, es_host: str = "http://localhost:9200"):
        self.es = Elasticsearch(es_host)

    def list_indices(self) -> list[str]:
        return [alias for alias in self.es.indices.get_alias().keys() if not alias.startswith(".")]

    def delete_index(self, index_name):
        self.es.indices.delete(index=index_name)

    def create_index(self, index_name: str):
        """Create a new index using a mapping file (`schema/es/products.json`).

        Args:
            index_name (str): An index name to create.
        """
        with open("schema/es/products.json") as file:
            schema = json.load(file)
            print(schema)
        self.es.indices.create(index=index_name, mappings=schema["mappings"])

    def count_docs(self, index_name: str) -> int:
        return self.es.count(index=index_name)["count"]

    def index_doc(self, index_name: str, doc: dict[str, Any], doc_id: Optional[str] = None):
        self.es.index(index=index_name, document=doc, id=doc_id)
        self.es.indices.refresh(index=index_name)

    @staticmethod
    def _generate_actions(
        index_name: str,
        docs: list[dict[str, Any]],
        id_fn: Optional[Callable[[dict[str, Any]], str]] = None,
    ) -> Iterator[dict[str, Any]]:
        """Convert docs to a bulk action iterator.

        ```
        {
            "product_id": "1", "product_title": "product title"
        }
        # The above doc will be as follows:
        {
            "_op_type": "index",
            "_index": "your-index-name",
            "_id": "1",
            "_source": {
                "product_id": "1",
                "product_title": "product title",
            },
        }
        ```

        Args:
            index_name (str): The Elasticsearch index name to index.
            docs (list[dict[str, Any]]): Docs to convert
            id_fn (Optional[Callable[[dict[str, Any]], str]], optional): If given, `id` is extracted and added to `_id`.
                Defaults to None == `_id` is not added.

        Yields:
            Iterator[dict[str, Any]]: Generated bulk actions
        """
        for doc in docs:
            action: dict[str, Any] = {
                "_op_type": "index",
                "_index": index_name,
            }
            if id_fn:
                action["_id"] = id_fn(doc)
            action["_source"] = doc
            yield action

    def index_docs(
        self,
        index_name: str,
        docs: list[dict[str, Any]],
        id_fn: Optional[Callable[[dict[str, Any]], str]] = None,
    ) -> tuple[int, int | list[dict[str, Any]]]:
        return helpers.bulk(
            client=self.es,
            actions=self._generate_actions(index_name, docs, id_fn),
            stats_only=True,
            raise_on_error=False,
        )

    @staticmethod
    def _convert_es_response_to_response(es_response: Any) -> Response:
        """Map a raw Elasticsearch response to our Response class for convenience.

        Args:
            es_response (Any): An Elasticsearch response to convert.

        Returns:
            Response: Our Response object.
        """
        return Response(
            results=[
                Result(
                    product=hit["_source"],
                    score=hit["_score"],
                    explanation=hit.get("_explanation", None),
                )
                for hit in es_response["hits"]["hits"]
            ],
            total_hits=es_response["hits"]["total"]["value"],
        )

    def analyze(self, text: str) -> dict[str, Any]:
        return self.es.indices.analyze(text=text).body

    def search(
        self,
        index_name: str,
        query: dict[str, Any],
        knn_query: Optional[dict[str, Any]] = None,
        size: int = 20,
        explain: bool = False,
    ) -> Response:
        """Perform a search and return a Response object.

        Args:
            index_name (str): An index name to perform a search.
            query (dict[str, Any]): an Elasticsearch query represented as a dict.
            knn_query (Optional[dict[str, Any]], optional): A knn clause to perform a KNN search. Defaults to None.
            size (int, optional): The number of hits to return. Defaults to 20.
            explain (bool, optional): If True, returns detailed information about score computation
                as part of a hit if True. Defaults to False.

        Returns:
            Response: A Response object.
        """
        es_response = self.es.search(index=index_name, query=query, knn=knn_query, size=size, explain=explain)
        return self._convert_es_response_to_response(es_response)

    def knn_search(self, index_name: str, knn_query: dict[str, Any]) -> Response:
        es_response = self.es.knn_search(index=index_name, knn=knn_query)
        return self._convert_es_response_to_response(es_response)

    def close(self):
        self.es.close()
