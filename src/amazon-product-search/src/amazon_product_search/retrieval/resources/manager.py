import logging
import threading
from typing import Any, Dict, Union

import torch
from transformers import AutoModel, AutoTokenizer

from amazon_product_search.constants import HF
from amazon_product_search.modules.colbert import ColBERTer
from amazon_product_search.modules.splade import Splade
from amazon_product_search.retrieval.core.protocols import ResourceManager
from dense_retrieval.encoders import SBERTEncoder

logger = logging.getLogger(__name__)


# Type alias for different encoder types
EncoderType = Union[ColBERTer, Splade, SBERTEncoder]


class SharedResourceManager(ResourceManager):
    """Manages shared resources like models, encoders, and tokenizers."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Singleton pattern to ensure shared resources
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._encoders: Dict[str, EncoderType] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._initialized = True

        logger.info(f"Initialized ResourceManager with device: {self._device}")

    def get_encoder(self, model_name: str) -> Any:
        """Get a cached encoder instance."""
        with self._lock:
            if model_name not in self._encoders:
                logger.info(f"Loading encoder for model: {model_name}")

                encoder: EncoderType
                if model_name in [HF.JP_COLBERT]:
                    # ColBERT encoder
                    encoder = ColBERTer("cl-tohoku/bert-base-japanese-v2")
                    encoder.load_state_dict(torch.load(model_name, map_location=self._device))
                    encoder.eval()
                    encoder.to(self._device)
                elif model_name in [HF.JP_SPLADE]:
                    # SPLADE encoder
                    encoder = Splade("cl-tohoku/bert-base-japanese-v2")
                    encoder.load_state_dict(torch.load(model_name, map_location=self._device))
                    encoder.eval()
                    encoder.to(self._device)
                else:
                    # Dense retrieval encoder (SBERT-style)
                    encoder = SBERTEncoder(model_name)

                self._encoders[model_name] = encoder
                logger.info(f"Loaded and cached encoder for: {model_name}")

            return self._encoders[model_name]

    def get_tokenizer(self, model_name: str) -> Any:
        """Get a cached tokenizer instance."""
        with self._lock:
            if model_name not in self._tokenizers:
                logger.info(f"Loading tokenizer for model: {model_name}")

                # Determine the base model name for tokenizer
                if model_name in [HF.JP_COLBERT, HF.JP_SPLADE]:
                    base_model = "cl-tohoku/bert-base-japanese-v2"
                else:
                    base_model = model_name

                tokenizer = AutoTokenizer.from_pretrained(base_model)
                self._tokenizers[model_name] = tokenizer
                logger.info(f"Loaded and cached tokenizer for: {model_name}")

            return self._tokenizers[model_name]

    def get_model(self, model_name: str) -> Any:
        """Get a cached raw model instance."""
        with self._lock:
            if model_name not in self._models:
                logger.info(f"Loading model: {model_name}")
                model = AutoModel.from_pretrained(model_name)
                model.eval()
                model.to(self._device)
                self._models[model_name] = model
                logger.info(f"Loaded and cached model: {model_name}")

            return self._models[model_name]

    def get_device(self) -> torch.device:
        """Get the device being used for computation."""
        return self._device

    def preload_models(self, model_names: list[str]) -> None:
        """Preload a list of models to avoid lazy loading during requests."""
        logger.info(f"Preloading {len(model_names)} models")
        for model_name in model_names:
            try:
                self.get_encoder(model_name)
                self.get_tokenizer(model_name)
                logger.info(f"Preloaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")

    def get_memory_usage(self) -> dict[str, Any]:
        """Get memory usage information."""
        memory_info = {
            "num_encoders": len(self._encoders),
            "num_tokenizers": len(self._tokenizers),
            "num_models": len(self._models),
            "device": str(self._device)
        }

        if torch.cuda.is_available():
            memory_info["cuda_memory_allocated"] = torch.cuda.memory_allocated()
            memory_info["cuda_memory_cached"] = torch.cuda.memory_reserved()

        return memory_info

    def cleanup(self) -> None:
        """Clean up all cached resources."""
        logger.info("Cleaning up ResourceManager")

        with self._lock:
            # Move models to CPU and clear cache
            for model in self._models.values():
                if hasattr(model, "cpu"):
                    model.cpu()

            for encoder in self._encoders.values():
                if hasattr(encoder, "cpu"):
                    encoder.cpu()

            self._encoders.clear()
            self._tokenizers.clear()
            self._models.clear()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("ResourceManager cleanup completed")

    def list_cached_resources(self) -> dict[str, list[str]]:
        """List all currently cached resources."""
        with self._lock:
            return {
                "encoders": list(self._encoders.keys()),
                "tokenizers": list(self._tokenizers.keys()),
                "models": list(self._models.keys())
            }
