import os

PROJECT_DIR = os.getenv("PROJECT_DIR", ".")
DATA_DIR = os.getenv("DATA_DIR", "data")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
VERTEX_DIR = os.getenv("VERTEX_DIR", "vertex")
VESPA_DIR = os.getenv("VESPA_DIR", "vespa")

PROJECT_ID = os.getenv("PROJECT_ID", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "")
REGION = os.getenv("REGION", "asia-northeast1")
STAGING_BUCKET = os.getenv("STAGING_BUCKET", "")
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT", "")

# Container Registry
TRAINING_IMAGE_URI = f"gcr.io/{PROJECT_ID}/{PROJECT_NAME}/training"
INDEXING_IMAGE_URI = f"gcr.io/{PROJECT_ID}/{PROJECT_NAME}/indexing"

# BigQuery
DATASET_ID = "amazon"


class HF:
    # For English
    EN_ELECTRA = "cross-encoder/ms-marco-electra-base"
    EN_MSMARCO = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    EN_ROBERTA = "sentence-transformers/msmarco-roberta-base-v3"
    EN_MULTIQA = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # 768D
    EN_ALL_MINI = "sentence-transformers/all-MiniLM-L6-v2"  # 384D
    EN_FINE_TUNED_ALL_MINI = f"{MODELS_DIR}/all-MiniLM-L6-v2"  # 384D

    # For Japanese
    JP_BERT = "cl-tohoku/bert-base-japanese-v2"  # 768D
    JP_SBERT_MEAN = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # 768D
    JP_SLUKE_MEAN = "sonoisa/sentence-luke-japanese-base-lite"  # 768D
    JP_DISTILBERT = "line-corporation/line-distilbert-base-japanese"
    JP_DEBERTA = "ku-nlp/deberta-v2-base-japanese"
    JP_FINE_TUNED_SBERT = f"{MODELS_DIR}/jp_fine_tuned_sbert"
    JP_COLBERT = f"{MODELS_DIR}/jp_colberter.pt"
    JP_SPLADE = f"{MODELS_DIR}/jp_splade.pt"

    # Multi-lingual
    ML_MPNET = "paraphrase-multilingual-mpnet-base-v2"
    ML_XLM = "stsb-xlm-r-multilingual"
