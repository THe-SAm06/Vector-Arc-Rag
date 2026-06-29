import numpy as np
import logging
from typing import List, Union
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install dependencies: pip install sentence-transformers numpy torch")

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Singleton-patterned Embedding Engine to prevent redundant model loading in memory.
    Transforms raw text into semantic dense vectors.
    """
    _instance = None

    def __new__(cls, model_name: str = 'all-MiniLM-L6-v2'):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            cls._instance._initialize_model(model_name)
        return cls._instance

    def _initialize_model(self, model_name: str):
        """Loads the HuggingFace model and automatically detects hardware acceleration."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing Embedding Engine '{model_name}' on device: {device.upper()}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}. Error: {str(e)}")
            raise

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates normalized embeddings for one or multiple strings.
        Normalization is enforced to allow fast dot-product Cosine Similarity later.
        """
        try:
            # We normalize embeddings here so Cosine Similarity becomes a simple dot product
            embeddings = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise