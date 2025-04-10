import logging
import os

import pandas as pd
from data_source import Locale
from google.cloud import bigquery
from pandas import DataFrame

from amazon_product_search.constants import DATA_DIR, DATASET_ID, PROJECT_ID


class QueryVectorCache:
    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def load(
        self, locale: Locale, project_id: str = PROJECT_ID, dataset_id: str = DATASET_ID, data_dir: str = DATA_DIR
    ) -> None:
        """Attempt to load query vector cache from file, otherwise load from BigQuery.

        Args:
            locale (Locale): The locale to load the cache for.
            project_id (str, optional): The BigQuery project ID. Defaults to PROJECT_ID.
            dataset_id (str, optional): The dataset ID. Defaults to DATASET_ID.
            data_dir (str, optional): The data directory to save and load the cache from. Defaults to DATA_DIR.
        """
        self._cache = self._load_cache_from_file(locale, data_dir)
        if self._cache:
            return
        self._cache = self._load_cache_from_bq(locale, project_id, dataset_id, data_dir)

    def _load_cache_from_file(self, locale: Locale, data_dir: str) -> dict[str, list[float]]:
        filepath = f"{data_dir}/query_vector_cache_{locale}.parquet"
        if not os.path.isfile(filepath):
            logging.info(f"Attempted to load query vector cache from {filepath} but file does not exist.")
            return {}
        df = pd.read_parquet(filepath)
        df["query_vector"] = df["query_vector"].apply(lambda v: v.tolist())
        logging.info(f"Query vector cache loaded from {filepath} with {len(df)} rows.")
        return self._df_to_cache_dict(df)

    def _load_cache_from_bq(
        self, locale: Locale, project_id: str, dataset_id: str, data_dir: str
    ) -> dict[str, list[float]]:
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
            df = bigquery.Client().query(sql).to_dataframe()
            logging.info(f"Query vector cache loaded from BigQuery with {len(df)} rows.")
            self._save_cache_to_file(df, locale, data_dir)
            cache = self._df_to_cache_dict(df)
        except Exception as e:
            logging.error(e)
        return cache

    def _save_cache_to_file(self, df: DataFrame, locale: Locale, data_dir: str) -> None:
        filepath = f"{data_dir}/query_vector_cache_{locale}.parquet"
        df.to_parquet(filepath)
        logging.info(f"Query vector cache saved to {filepath}")

    def _df_to_cache_dict(self, df: DataFrame) -> dict[str, list[float]]:
        cache: dict[str, list[float]] = {}
        for row in df.to_dict(orient="records"):
            cache[row["query"]] = row["query_vector"]
        return cache

    def __getitem__(self, query: str) -> list[float] | None:
        return self._cache.get(query, None)
