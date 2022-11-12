import os

DATA_DIR = os.getenv("DATA_DIR", "data")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
PROJECT_ID = os.getenv("PROJECT_ID", "")
PROJECT_NAME = os.getenv("PROJECT_NAME", "")
REGION = os.getenv("REGION", "asia-northeast1")

IMAGE_NAME = "indexer"
IMAGE_URI = f"gcr.io/{PROJECT_ID}/{PROJECT_NAME}/{IMAGE_NAME}"
