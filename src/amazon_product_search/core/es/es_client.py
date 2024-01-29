import json
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel

from amazon_product_search.constants import MODELS_DIR
from amazon_product_search.core.retrieval.response import Response, Result
from amazon_product_search.core.source import Locale
from elasticsearch import Elasticsearch, helpers


class EsClient:
    """A wrapper class of https://elasticsearch-py.readthedocs.io/"""

    def __init__(self, es_host: str = "http://localhost:9200") -> None:
        self.es = Elasticsearch(es_host)

    def list_indices(self) -> list[str]:
        return [alias for alias in self.es.indices.get_alias() if not alias.startswith(".")]

    def delete_index(self, index_name: str) -> None:
        self.es.indices.delete(index=index_name)

    def create_index(self, locale: Locale, index_name: str) -> None:
        """Create a new index using a mapping file (`elasticsearch/schemas/products_{locale}.json`).

        Args:
            index_name (str): An index name to create.
        """
        with open(f"elasticsearch/schemas/products_{locale}.json") as file:
            schema = json.load(file)
            print(schema)
        self.es.indices.create(index=index_name, settings=schema.get("settings"), mappings=schema.get("mappings"))

    def import_model(self, model_id: str, tmp_path: str = MODELS_DIR) -> None:
        """Import a TransformerModel into Elasticsearch.

        Call `POST _ml/trained_models/_model_id_/deployment/_start` after the model is imported.
        The imported model can be used as follows:
        ```
        GET products_jp/_search
        {
          "query": {
            "match": {
              "product_title": {
                "query": "東京"
              }
            }
          },
          "knn": [
            {
              "field": "product_vector",
              "k": 10,
              "num_candidates": "100",
              "query_vector_builder": {
                "text_embedding": {
                  "model_id": "sonoisa__sentence-bert-base-ja-mean-tokens-v2",
                  "model_text": "東京"
                }
              }
            }
          ]
        }
        ```

        Args:
            model_id (str): The name of the transformer model to import.
            tmp_path (str, optional): The directory to save the model. Defaults to MODELS_DIR.
        """
        tm = TransformerModel(model_id, task_type="text_embedding")

        Path(tmp_path).mkdir(parents=True, exist_ok=True)
        model_path, config, vocab_path = tm.save(tmp_path)

        ptm = PyTorchModel(self.es, tm.elasticsearch_model_id())
        ptm.import_model(
            model_path=model_path,
            config_path=None,
            vocab_path=vocab_path,
            config=config,
        )
        ptm.start()

    def count_docs(self, index_name: str) -> int:
        return self.es.count(index=index_name)["count"]

    def index_doc(self, index_name: str, doc: dict[str, Any], doc_id: Optional[str] = None) -> None:
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

    def get_termvectors(self, index_name: str, fields: list[str], text: str) -> dict[str, Any]:
        """Return term vectors of the given text.

        This function call the following API:
        https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-termvectors.html

        An example response is as follows:

        ```
        {
          "_index": "products_jp",
          "_version": 0,
          "found": true,
          "took": 0,
          "term_vectors": {
            "product_title": {
              "field_statistics": {
                "sum_doc_freq": 7414978,
                "doc_count": 339055,
                "sum_ttf": 8573921
              },
              "terms": {
                "nike": {
                  "doc_freq": 420,
                  "ttf": 424,
                  "term_freq": 1
                }
              }
            }
          }
        }
        ```

        Args:
            index_name (str): An index name to analyze.
            text (str): A text to analyze.

        Returns:
            dict[str, Any]: A term vector response.
        """
        doc = {field: text for field in fields}
        es_response = self.es.termvectors(
            index=index_name,
            doc=doc,
            term_statistics=True,
            field_statistics=True,
            positions=False,
            offsets=False,
        )
        return es_response["term_vectors"]

    def search(
        self,
        index_name: str,
        query: dict[str, Any] | None = None,
        knn_query: dict[str, Any] | None = None,
        rank: dict[str, Any] | None = None,
        size: int = 20,
        explain: bool = False,
        request_cache: bool | None = None,
    ) -> Response:
        """Perform a search and return a Response object.

        Args:
            index_name (str): An index name to perform a search.
            query (Optional[dict[str, Any]], optional): an Elasticsearch query represented as a dict.
            knn_query (Optional[dict[str, Any]], optional): A knn clause to perform a KNN search. Defaults to None.
            rank (Optional[dict[str, Any]], optional): A rank clause to perform a rank fusion. Defaults to None.
            size (int, optional): The number of hits to return. Defaults to 20.
            explain (bool, optional): If True, returns detailed information about score computation
                as part of a hit if True. Defaults to False.
            request_cache (bool | None, optional): If True, the request is cached. Defaults to None.

        Returns:
            Response: A Response object.
        """
        es_response = self.es.search(
            index=index_name,
            query=query,
            knn=knn_query,
            rank=rank,
            size=size,
            explain=explain,
            request_cache=request_cache,
        )
        return self._convert_es_response_to_response(es_response)

    def close(self) -> None:
        self.es.close()
