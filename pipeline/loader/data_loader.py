import re
from typing import List
from docling.document import Document
from docling.serializers import get_serializer

class DoclingLoader:
    def __init__(
        self,
        pdf_path: str,
        tokenizer,
        max_tokens: int = 300,
        merge_peers: bool = True,
        mode: str = "hybrid",
        serializer_provider: str = "default",
        min_token_threshold: int = 20,
        chunk_size: int = 256,
        chunk_overlap: int = 32
    ):
        self.pdf_path = pdf_path
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.mode = mode
        self.serializer_provider = serializer_provider
        self.min_token_threshold = min_token_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\x0c\r]", "", text)
        text = text.strip()
        return text

    def _filter_chunks(self, chunks: List[str]) -> List[str]:
        return [
            chunk for chunk in chunks
            if len(self.tokenizer(chunk)) >= self.min_token_threshold
        ]

    def _chunk_with_overlap(self, text: str) -> List[str]:
        tokens = self.tokenizer(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens) if hasattr(self.tokenizer, "decode") else " ".join(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def load(self) -> List[str]:
        document = Document.from_pdf(
            self.pdf_path,
            max_tokens=self.max_tokens,
            merge_peers=self.merge_peers,
            mode=self.mode,
        )

        serializer = get_serializer(self.serializer_provider)
        raw_chunks = [self._clean_text(section["text"]) for section in serializer.serialize(document)]

        filtered = self._filter_chunks(raw_chunks)

        final_chunks = []
        for chunk in filtered:
            if len(self.tokenizer(chunk)) > self.chunk_size:
                final_chunks.extend(self._chunk_with_overlap(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

