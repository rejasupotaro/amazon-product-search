from typing import Any, Callable, Dict, Optional

from requests.models import Response

import amazon_product_search.vespa.service as vespa_service
from vespa.application import ApplicationPackage
from vespa.io import VespaQueryResponse, VespaResponse


class VespaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        app_package: Optional[ApplicationPackage] = None,
    ):
        self.vespa_app = vespa_service.connect(host, app_package)

    def feed(
        self,
        schema: str,
        docs: list[dict[str, Any]],
        id_fn: Callable[[Dict[str, Any]], str],
    ) -> list[VespaResponse]:
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
        print(batch)
        return self.vespa_app.feed_batch(schema=schema, batch=batch)

    def search(self, query: Optional[dict[str, Any]] = None) -> VespaQueryResponse:
        """Send a search request to Vespa.

        Args:
            query (Optional[str], optional): A dict containing all the request params.

        Returns:
            VespaQueryResponse: A response from Vespa.
        """
        return self.vespa_app.query(query)

    def delete_all_docs(self, content_cluster_name: str, schema: str) -> Response:
        """Delete all docs associated with the given schema.

        Args:
            content_cluster_name (str): The name of the content cluster.
            schema (str): The schema that we are deleting data from.

        Returns:
            Response: The response of the HTTP DELETE request.
        """
        return self.vespa_app.delete_all_docs(content_cluster_name, schema)
