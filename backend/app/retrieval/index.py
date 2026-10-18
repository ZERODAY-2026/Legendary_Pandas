"""
BM25 Inverted Index over document chunks.
Strictly local sparse retrieval without external embedding models.
"""

import math
import re
from backend.app.document.schemas import Chunk

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # Minimal pure Python fallback for BM25Okapi if package is not installed
    class BM25Okapi:  # type: ignore
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
            self.doc_freqs: list[dict[str, int]] = []
            self.idf: dict[str, float] = {}
            self.doc_len: list[int] = []
            self.k1 = k1
            self.b = b

            nd: dict[str, int] = {}
            for document in corpus:
                self.doc_len.append(len(document))
                frequencies: dict[str, int] = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1

            for word, freq in nd.items():
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        def get_scores(self, query: list[str]) -> list[float]:
            scores = [0.0] * self.corpus_size
            for q in query:
                q_idf = self.idf.get(q, 0.0)
                for i, doc_freq in enumerate(self.doc_freqs):
                    freq = doc_freq.get(q, 0)
                    denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl)) if self.avgdl > 0 else 1
                    scores[i] += q_idf * (freq * (self.k1 + 1)) / denom
            return scores


class BM25DocumentIndex:
    """In-memory BM25 index built over the chunks of the active document."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())
