"""
cold_storage.py
───────────────
BM25-based cold storage backend.

This is the default (and currently only) implementation of BaseColdStorage.
It uses the BM25Okapi statistical ranking algorithm to retrieve documents
from the SciFact benchmark corpus.

Why BM25 for the demo?
  • Zero vector embedding overhead — retrieval is pure term frequency math
  • Deterministic and reproducible (no model dependencies)
  • Sub-millisecond lookup on a 500-doc corpus
  • Straightforward to swap out for SQL or a vector DB later

Swap strategy:
  Replace this by passing cold_storage=YourBackend() to AdaptiveRAGSystem.
  See base_cold_storage.py for the interface contract.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.base_cold_storage import BaseColdStorage

logger = logging.getLogger(__name__)


class BM25ColdStorage(BaseColdStorage):
    """
    Production-grade text fallback using BM25Okapi ranking.

    Indexed once at startup; all subsequent queries are O(N·|vocab|) lookups
    where N is corpus size. For the demo corpus (500 docs) this is
    effectively instantaneous (< 1ms per query).
    """

    def __init__(self, data_path: str = "data/scifact_corpus.json"):
        self.data_path = data_path
        self.corpus_keys: List[str] = []
        self.corpus_texts: List[str] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_and_index()

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase alphanumeric tokenizer; strips common punctuation noise."""
        return [
            word.strip(".,!?\"'()[]{}<>:;-").lower()
            for word in text.split()
            if word
        ]

    def _load_and_index(self) -> None:
        """Loads the corpus JSON and fits the BM25 index. Called once at init."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Corpus not found at '{self.data_path}'. "
                "Run src/data_ingestion.py first to download the SciFact dataset."
            )

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data: Dict[str, str] = json.load(f)

        if not raw_data:
            raise ValueError(f"Corpus at '{self.data_path}' is empty.")

        for doc_id, text in raw_data.items():
            self.corpus_keys.append(doc_id)
            self.corpus_texts.append(text)

        tokenized_corpus = [self._tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(
            f"💾 BM25ColdStorage: indexed {len(self.corpus_keys)} documents "
            f"from '{self.data_path}'"
        )

    # ──────────────────────────────────────────────────────────────
    # BaseColdStorage interface
    # ──────────────────────────────────────────────────────────────

    def search(self, query: str) -> Optional[Tuple[str, str]]:
        """
        BM25 retrieval — finds the best matching document for a query.

        Returns (doc_id, text) of the highest-scoring document,
        or None if no terms match (score == 0).
        """
        if not self.bm25:
            logger.error("BM25 index was never built — call _load_and_index() first.")
            return None

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        best_idx = int(scores.argmax())

        if scores[best_idx] <= 0.0:
            logger.warning(
                f"No keyword overlap found in corpus for query: '{query[:50]}...'"
            )
            return None

        return self.corpus_keys[best_idx], self.corpus_texts[best_idx]

    def get_corpus_size(self) -> int:
        return len(self.corpus_keys)


# ──────────────────────────────────────────────────────────────────────────────
# Backwards-compatibility alias
# Existing code that imports ProductionColdStorage continues to work unchanged.
# ──────────────────────────────────────────────────────────────────────────────
ProductionColdStorage = BM25ColdStorage