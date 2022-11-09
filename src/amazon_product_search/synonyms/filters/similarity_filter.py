import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from amazon_product_search.nlp.encoder import Encoder


class SimilarityFilter:
    def __init__(self, batch_size: int = 8):
        self.encoder = Encoder()
        self.batch_size = batch_size

    def calculate_score(self, left: list[str], right: list[str]) -> list[float]:
        left_tensor = self.encoder.encode(left, convert_to_tensor=True)
        right_tensor = self.encoder.encode(right, convert_to_tensor=True)
        return F.cosine_similarity(left_tensor, right_tensor, dim=1).tolist()

    def apply(self, synonyms_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Filter out synonyms based on the similarity between terms.

        Args:
            syonyms_df (pd.DataFrame): A dataframe that contains synonym pairs.
            threshold (float):
        Returns:
            The filtered dataframe.
        """
        scores = []
        for batch in tqdm(np.array_split(synonyms_df, len(synonyms_df) // self.batch_size)):
            queries = batch["query"].tolist()
            titles = batch["title"].tolist()
            scores.extend(self.calculate_score(queries, titles))

        synonyms_df["similarity"] = scores
        synonyms_df = synonyms_df[synonyms_df["similarity"] > threshold]
        return synonyms_df.sort_values("similarity", ascending=False)
