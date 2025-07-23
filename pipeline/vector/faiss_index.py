import os
import faiss
import numpy as np
import pickle
from typing import List, Optional


class FAISSEngine:
    def __init__(self, dim: int, index_path: Optional[str] = None):

        self.dim = dim
        self.index_path = index_path or "faiss_index.index"
        self.meta_path = self.index_path.replace(".index", "_meta.pkl")

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load_index()
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.id_to_text = {}

    def build_index(self, embeddings: List[List[float]], texts: List[str]):
        """
        Build a new FAISS index from given embeddings and associate with text chunks.

        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            texts (List[str]): Original text chunks corresponding to embeddings.
        """
        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)
        self.id_to_text = {i: text for i, text in enumerate(texts)}

    def save_index(self):
        """Save FAISS index and associated text metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.id_to_text, f)

    def _load_index(self):
        """Internal method to load FAISS index and metadata."""
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.id_to_text = pickle.load(f)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        Perform a similarity search on the FAISS index.

        Args:
            query_embedding (List[float]): Embedding vector of the query.
            top_k (int): Number of top results to return.

        Returns:
            List[str]: Top matching text chunks.
        """
        query = np.array([query_embedding]).astype("float32")
        _, indices = self.index.search(query, top_k)
        return [self.id_to_text[i] for i in indices[0] if i in self.id_to_text]
