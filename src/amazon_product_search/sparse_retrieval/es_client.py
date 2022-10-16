import json
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from elasticsearch import Elasticsearch, helpers


class EsClient:
    def __init__(self, es_host: str):
        self.es = Elasticsearch(es_host)

    def list_indices(self) -> List[str]:
        return [alias for alias in self.es.indices.get_alias().keys() if not alias.startswith(".")]

    def delete_index(self, index_name):
        self.es.indices.delete(index=index_name)

    def create_index(self, index_name: str):
        with open("schema/es/products.json") as file:
            mappings = json.load(file)
            print(mappings)
        self.es.indices.create(index=index_name, mappings=mappings)

    def count_docs(self, index_name: str) -> Any:
        return self.es.count(index=index_name)

    def index_doc(self, index_name: str, doc: Dict[str, Any]):
        self.es.index(index=index_name, document=doc)
        self.es.indices.refresh(index="products_jp")

    def generate_actions(
        self,
        index_name: str,
        docs: List[Dict[str, Any]],
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        for doc in docs:
            action: Dict[str, Any] = {
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
        docs: List[Dict[str, Any]],
        id_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> Tuple[int, Union[int, List[Dict[str, Any]]]]:
        return helpers.bulk(
            client=self.es,
            actions=self.generate_actions(index_name, docs, id_fn),
            stats_only=True,
            raise_on_error=False,
        )

    def search(self, index_name: str, es_query: Dict[str, Any], size: int = 20) -> Any:
        response = self.es.search(index=index_name, query=es_query, size=size)
        return response

    def close(self):
        self.es.close()
