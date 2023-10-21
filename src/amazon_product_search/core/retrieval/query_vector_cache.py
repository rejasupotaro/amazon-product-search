import logging

from google.cloud import bigquery

from amazon_product_search.constants import DATASET_ID, PROJECT_ID
from amazon_product_search.core.source import Locale


class QueryVectorCache:
    def __init__(self):
        self._cache = {}

    def load(self, locale: Locale, project_id: str = PROJECT_ID, dataset_id: str = DATASET_ID) -> None:
        self._cache = self._load_cache_from_bq(locale, project_id, dataset_id)

    def _load_cache_from_bq(self, locale: Locale, project_id: str, dataset_id: str) -> dict[str, list[float]]:
        sql = f"""
        SELECT
            query,
            query_vector,
        FROM
            `{project_id}.{dataset_id}.queries_{locale}`
        LIMIT
            1000000
        """
        cache: dict[str, list[float]] = {}
        try:
            rows_df = bigquery.Client().query(sql).to_dataframe()
            for row in rows_df.to_dict(orient="records"):
                cache[row["query"]] = row["query_vector"]
        except Exception as e:
            logging.error(e)
        return cache

    def __getitem__(self, query: str) -> list[float] | None:
        return self._cache.get(query, None)
