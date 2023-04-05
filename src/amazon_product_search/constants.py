import os

DATA_DIR = os.getenv("DATA_DIR", "data")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
VESPA_DIR = os.getenv("VESPA_DIR", "vespa")
PROJECT_ID = os.getenv("PROJECT_ID", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "")
REGION = os.getenv("REGION", "asia-northeast1")


class HF:
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
    # JA_ROBERTA = "nlp-waseda/roberta-large-japanese"
    JA_FINE_TUNED_SBERT = f"{MODELS_DIR}/jp_fine_tuned_sbert"
    JP_COLBERT = f"{MODELS_DIR}/jp_colberter.pt"
    JP_SPLADE = f"{MODELS_DIR}/jp_splade.pt"

    # Multi-lingual
    # ML_MPNET = "paraphrase-multilingual-mpnet-base-v2"
    # ML_XLM = "stsb-xlm-r-multilingual"
