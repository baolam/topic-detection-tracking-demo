import numpy as np
import hashlib
from typing import List

class TextEmbedder:
    """Real-time text embedding extractor with SentenceTransformers or fast semantic hash fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_dim: int = 384):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.vector_dim = self.model.get_sentence_embedding_dimension()
            print(f"[Embedder] Successfully loaded SentenceTransformer: {self.model_name}")
        except Exception as e:
            print(f"[Embedder] Notice: SentenceTransformer model loading fallback to Hash Embedding ({e})")
            self.model = None

    def encode(self, text: str) -> np.ndarray:
        """Encodes text into a normalized 1D float32 vector."""
        if self.model is not None:
            vec = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return vec.astype(np.float32)
        else:
            return self._hash_encode(text)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encodes a list of texts into a 2D float32 numpy matrix."""
        if self.model is not None:
            vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return vecs.astype(np.float32)
        else:
            return np.vstack([self._hash_encode(t) for t in texts])

    def _hash_encode(self, text: str) -> np.ndarray:
        """Fast fallback semantic hash feature generator (normalized)."""
        words = text.lower().split()
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        for w in words:
            # Hash word to vector index
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % self.vector_dim
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec = np.ones(self.vector_dim, dtype=np.float32) / np.sqrt(self.vector_dim)
        return vec
