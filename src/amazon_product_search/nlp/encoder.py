from sentence_transformers import SentenceTransformer
from torch import Tensor

from amazon_product_search.constants import MODELS_DIR

# For English
# EN_ELECTRA = "cross-encoder/ms-marco-electra-base"
# EN_MSMARCO = "cross-encoder/ms-marco-MiniLM-L-12-v2"
# EN_ROBERTA = "sentence-transformers/msmarco-roberta-base-v3"

# For Spanish
# ES_BERT = "dccuchile/bert-base-spanish-wwm-uncased"
# ES_ROBERTA = "bertin-project/bertin-roberta-base-spanish"

# For Japanese
JA_BERT = "cl-tohoku/bert-base-japanese-v2"
JA_SBERT = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
JA_FINE_TUNED_SBERT = f"{MODELS_DIR}/jp_fine_tuned_sbert"
JA_COLBERT = f"{MODELS_DIR}/jp_colberter.pt"
# JA_ROBERTA = "nlp-waseda/roberta-large-japanese"

# Multi-lingual
# ML_MPNET = "paraphrase-multilingual-mpnet-base-v2"
# ML_XLM = "stsb-xlm-r-multilingual"


class Encoder:
    def __init__(self, model_name: str = JA_SBERT):
        self.embedder = SentenceTransformer(model_name)

    def encode(self, texts: list[str], show_progress_bar: bool = False, convert_to_tensor: bool = False) -> Tensor:
        return self.embedder.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
        )
