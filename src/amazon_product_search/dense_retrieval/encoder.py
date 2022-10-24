from sentence_transformers import SentenceTransformer
from torch import Tensor

TOHOKU = "cl-tohoku/bert-base-japanese-v2"
SONOISA = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"

class Encoder:
    def __init__(self, model_name: str = TOHOKU):
        # For English
        # model_name = "cross-encoder/ms-marco-electra-base"
        # model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        # model_name = "sentence-transformers/msmarco-roberta-base-v3"

        # For Spanish
        # model_name = "dccuchile/bert-base-spanish-wwm-uncased"
        # model_name = "bertin-project/bertin-roberta-base-spanish"

        # For Japanese
        # model_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
        # model_name = "cl-tohoku/bert-base-japanese-v2"
        # model_name = "cl-tohoku/bert-base-japanese-char-v2"
        # model_name = "nlp-waseda/roberta-large-japanese"

        # Multi-lingual
        # model_name = "paraphrase-multilingual-mpnet-base-v2"
        # model_name = "stsb-xlm-r-multilingual"

        self.embedder = SentenceTransformer(model_name)

    def encode(self, texts: list[str], show_progress_bar: bool = True, convert_to_tensor: bool = False) -> Tensor:
        return self.embedder.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )
