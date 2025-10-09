from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np

class EmbeddingEncoder:
    def __init__(self, model_name: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✓ Loaded embedding model on {device}")
    
    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a batch of texts into embeddings"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
