from typing import Any, Optional

from requests.models import Response
from vespa.application import ApplicationPackage
from vespa.io import VespaQueryResponse, VespaResponse

import amazon_product_search.vespa.service as vespa_service


class VespaClient:
    def __init__(self, host: Optional[str] = None, app_package: Optional[ApplicationPackage] = None):
        self.vespa_app = vespa_service.connect(host, app_package)

    def feed(self, schema: str, batch: list[dict[str, Any]]) -> list[VespaResponse]:
        """Feed a batch of data to Vespa.

        Args:
            schema (str): The name of the target schema.
            batch (list[dict[str, Any]]): A list of dicts containing data.
                The expected format of a dict is `{"id": doc_id, "fields": doc}`.

        Returns:
            list[VespaResponse]: A list of VespaResponses.
        """
        return self.vespa_app.feed_batch(schema=schema, batch=batch)

    def search(self, query: str, hits: int = 10) -> VespaQueryResponse:
        """Send a search request to Vespa.

        Args:
            query (str): A string query.
            hits (int, optional): The max number of docs to return. Defaults to 10.

        Returns:
            VespaQueryResponse: _description_
        """
        return self.vespa_app.query(
            body={
                "yql": "select * from sources * where userQuery()",
                "query": query,
                "type": "any",
                "hits": hits,
            }
        )

    def delete_all_docs(self, content_cluster_name: str, schema: str) -> Response:
        """Delete all docs associated with the given schema.

        Args:
            content_cluster_name (str): The name of the content cluster.
            schema (str): The schema that we are deleting data from.

        Returns:
            Response: The response of the HTTP DELETE request.
        """
        return self.vespa_app.delete_all_docs(content_cluster_name, schema)
