import os

DATA_DIR = os.environ["DATA_DIR"]
MODELS_DIR = os.environ["MODELS_DIR"]
PROJECT_ID = os.environ["PROJECT_ID"]
PROJECT_NAME = os.environ["PROJECT_NAME"]
REGION = os.environ["REGION"]

IMAGE_NAME = "dense-indexer"
IMAGE_URI = f"gcr.io/{PROJECT_ID}/{PROJECT_NAME}/{IMAGE_NAME}"
