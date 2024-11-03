from typing import Any, Callable, Dict, Optional

from vespa.application import ApplicationPackage
from vespa.io import VespaQueryResponse, VespaResponse

import amazon_product_search.core.vespa.service as vespa_service
from amazon_product_search.core.retrieval.response import Response, Result


class VespaClient:
    def __init__(
        self,
        host: str = "",
        app_package: ApplicationPackage | None = None,
    ) -> None:
        self.vespa_app = vespa_service.connect(host, app_package)

    def feed(
        self,
        schema: str,
        docs: list[dict[str, Any]],
        id_fn: Callable[[Dict[str, Any]], str],
        callback_fn: Callable[[VespaResponse, str], None] | None = None,
    ) -> None:
        """Feed a batch of data to Vespa.

        Args:
            schema (str): The name of the target schema.
            docs (list[dict[str, Any]]): A list of documents to index.
            id_fn (Callable[[Dict[str, Any]], str]): A function to convert docs to
                the expected format: `[{"id": doc_id, "fields": doc}]`.

        Returns:
            list[VespaResponse]: A list of VespaResponses.
        """
        batch = [{"id": id_fn(doc), "fields": doc} for doc in docs]
        self.vespa_app.feed_iterable(iter=batch, schema=schema, callback=callback_fn)

    @staticmethod
    def _convert_vespa_response_to_response(vespa_response: VespaQueryResponse) -> Response:
        """Map a raw Elasticsearch response to our Response class for convenience.

        Args:
            es_response (Any): An Elasticsearch response to convert.

        Returns:
            Response: Our Response object.
        """
        hits = vespa_response.hits
        return Response(
            results=[
                Result(
                    product=hit["fields"],
                    score=hit["relevance"],
                    explanation=hit["fields"].get("summaryfeatures", {}),
                )
                for hit in hits
            ],
            total_hits=vespa_response.json["root"]["fields"]["totalCount"],
        )

    def search(self, query: Optional[dict[str, Any]] = None) -> Response:
        """Send a search request to Vespa.

        Args:
            query (Optional[str], optional): A dict containing all the request params.

        Returns:
            Response: A Response object.
        """
        vespa_response = self.vespa_app.query(query)
        return self._convert_vespa_response_to_response(vespa_response)

    def delete_all_docs(self, content_cluster_name: str, schema: str) -> Any:
        """Delete all docs associated with the given schema.

        Args:
            content_cluster_name (str): The name of the content cluster.
            schema (str): The schema that we are deleting data from.

        Returns:
            Response: The response of the HTTP DELETE request.
        """
        return self.vespa_app.delete_all_docs(content_cluster_name, schema).json()
