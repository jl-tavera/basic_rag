import os
from typing import List, Union
from dotenv import load_dotenv

import openai
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        load_dotenv()
        self.model_name = model_name
        self.batch_size = batch_size

        if model_name.startswith("openai/"):
            self.provider = "openai"
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.model = model_name.split("/", 1)[-1]  # e.g., text-embedding-3-small
        else:
            self.provider = "hf"
            self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "hf":
            return self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)

        elif self.provider == "openai":
            results = []
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i:i + self.batch_size]
                response = openai.embeddings.create(
                    input=chunk,
                    model=self.model
                )
                embeddings = [e.embedding for e in response.data]
                results.extend(embeddings)
            return results

        else:
            raise ValueError("Unsupported provider")
