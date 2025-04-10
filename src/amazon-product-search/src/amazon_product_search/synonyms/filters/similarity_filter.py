import torch
from more_itertools import chunked
from pandas import DataFrame
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from amazon_product_search.constants import HF
from dense_retrieval.encoders import SBERTEncoder


class SimilarityFilter:
    def __init__(self, hf_model_name: str = HF.JP_SLUKE_MEAN, batch_size: int = 8) -> None:
        self.encoder: SBERTEncoder = SBERTEncoder(hf_model_name)
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

    def apply(self, synonyms_df: DataFrame, threshold: float = 0.5) -> DataFrame:
        """Filter out synonyms based on the similarity between terms.

        Args:
            syonyms_df (pd.DataFrame): A dataframe that contains synonym pairs.
            threshold (float): The threshold for the similarity score.
        Returns:
            The filtered dataframe.
        """
        scores = []
        chunks = chunked(synonyms_df.to_dict("records"), self.batch_size)
        for batch in tqdm(list(chunks)):
            queries = [row["query"] for row in batch]
            titles = [row["title"] for row in batch]
            scores.extend(self.calculate_score(queries, titles))

        synonyms_df["similarity"] = scores
        synonyms_df = synonyms_df[synonyms_df["similarity"] > threshold]
        synonyms_df = synonyms_df.sort_values(by="similarity", ascending=False)
        return synonyms_df
