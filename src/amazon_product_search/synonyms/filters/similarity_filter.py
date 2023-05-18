import polars as pl
import torch
from more_itertools import chunked
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from amazon_product_search.constants import HF
from amazon_product_search_dense_retrieval.encoders import Encoder, SBERTEncoder


class SimilarityFilter:
    def __init__(self, model_name: str = HF.JP_SBERT, batch_size: int = 8):
        self.encoder: Encoder = SBERTEncoder(model_name)
        self.batch_size = batch_size

    def calculate_score(self, left: list[str], right: list[str]) -> list[float]:
        """Calculate the cosine similarity of the given two inputs.

        Args:
            left (list[str]): A list of input texts.
            right (list[str]): Another list of input texts.

        Returns:
            list[float]: A list of scores.
        """
        left_tensor = torch.from_numpy(self.encoder.encode(left))
        right_tensor = torch.from_numpy(self.encoder.encode(right))
        return cosine_similarity(left_tensor, right_tensor, dim=1).tolist()

    def apply(self, synonyms_df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
        """Filter out synonyms based on the similarity between terms.

        Args:
            syonyms_df (pd.DataFrame): A dataframe that contains synonym pairs.
            threshold (float):
        Returns:
            The filtered dataframe.
        """
        scores = []
        chunks = chunked(synonyms_df.to_dicts(), len(synonyms_df) // self.batch_size)
        for batch in tqdm(list(chunks)):
            queries = [row["query"] for row in batch]
            titles = [row["title"] for row in batch]
            scores.extend(self.calculate_score(queries, titles))

        synonyms_df = (
            synonyms_df.with_columns(pl.Series(name="similarity", values=scores))
            .filter(pl.col("similarity") > threshold)
            .sort("similarity", reverse=True)
        )
        return synonyms_df
