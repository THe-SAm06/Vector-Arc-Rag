"""
hybrid_cold_storage.py
──────────────────────
Hybrid BM25 + FAISS cold storage with Reciprocal Rank Fusion (RRF).

Design rationale (2026 research consensus)
─────────────────────────────────────────
Neither BM25 (sparse) nor dense vector search (FAISS) alone is optimal:
  • BM25 excels at exact terminology, product IDs, rare tokens
  • FAISS excels at semantic intent, synonyms, paraphrase understanding
  • Hybrid + RRF consistently outperforms either alone (Pinecone blog, 2024;
    Elasticsearch hybrid docs, 2025; "Benchmarking hybrid retrieval" arxiv 2024)

Reciprocal Rank Fusion (RRF):
  score_rrf(d) = Σ  1 / (k + rank_i(d))
               i ∈ {bm25, faiss}

  where k=60 is the standard constant (Cormack et al., 2009).
  k=60 smooths the contribution of low-ranked documents and is robust
  across diverse corpora — no per-dataset tuning needed.

Why FAISS flat index (not HNSW)?
  For corpora up to ~10,000 documents, exhaustive flat search is faster
  than HNSW due to HNSW's graph-building overhead and memory indirection.
  With our 10–500 document corpora, flat search completes in <1ms.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.base_cold_storage import BaseColdStorage
from src.embedder import EmbeddingEngine

logger = logging.getLogger(__name__)

# RRF constant — standard value from Cormack et al. 2009
_RRF_K = 60


class HybridColdStorage(BaseColdStorage):
    """
    Hybrid retrieval combining BM25 (sparse) and FAISS-flat (dense)
    with Reciprocal Rank Fusion.

    Inherits from BaseColdStorage — drop-in replacement for BM25ColdStorage.
    Interface: search(query: str) -> Optional[Tuple[str, str]]

    The FAISS index is built lazily on the first search() call so that
    the embedder model (and GPU) are only loaded when actually needed.
    """

    def __init__(self, data_path: str = "data/iit_dharwad_corpus.json"):
        self.data_path = data_path
        self._corpus: Dict[str, str] = {}
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []

        # BM25 index — built at init (lightweight, CPU-only)
        self._bm25: Optional[BM25Okapi] = None

        # FAISS dense index — built lazily on first search()
        self._faiss_index = None       # faiss.IndexFlatIP
        self._doc_vectors: Optional[np.ndarray] = None  # (N, 384) float32

        self._load_corpus()
        self._build_bm25()
        logger.info(
            f"HybridColdStorage: loaded {len(self._corpus)} documents "
            f"from '{data_path}' | BM25 ready | FAISS will build on first query"
        )

    # ──────────────────────────────────────────────────────────────
    # BaseColdStorage interface
    # ──────────────────────────────────────────────────────────────

    def get_corpus_size(self) -> int:
        """Returns the total number of documents indexed in this backend."""
        return len(self._doc_ids)

    def search(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Find the most relevant document using hybrid BM25 + FAISS retrieval
        with Reciprocal Rank Fusion.

        Returns (doc_id, doc_text) of the best fused result, or None.
        """
        if not self._doc_ids:
            return None

        # Ensure FAISS index is ready
        if self._faiss_index is None:
            self._build_faiss()

        # ── BM25 ranking ─────────────────────────────────────────────────────
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)          # shape (N,)
        bm25_ranks  = self._scores_to_ranks(bm25_scores)     # 0 = best

        # ── FAISS dense ranking ───────────────────────────────────────────────
        embedder = EmbeddingEngine()                           # singleton
        q_vec = embedder.embed(query).reshape(1, -1).astype(np.float32)
        dense_scores, _ = self._faiss_index.search(q_vec, len(self._doc_ids))
        # dense_scores[0] = similarity scores in descending order (best first)
        # We need scores per doc_id in corpus order
        dense_scores_ordered = self._faiss_index.reconstruct_n(0, len(self._doc_ids))
        # Use a simpler approach: compute dot products directly (index is flat)
        dense_dot = (self._doc_vectors @ q_vec.T).squeeze()  # shape (N,)
        dense_ranks = self._scores_to_ranks(dense_dot)        # 0 = best

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        rrf_scores = np.zeros(len(self._doc_ids), dtype=np.float64)
        for i in range(len(self._doc_ids)):
            rrf_scores[i] = (
                1.0 / (_RRF_K + bm25_ranks[i] + 1) +
                1.0 / (_RRF_K + dense_ranks[i] + 1)
            )

        best_idx = int(np.argmax(rrf_scores))
        doc_id   = self._doc_ids[best_idx]
        doc_text = self._doc_texts[best_idx]

        logger.debug(
            f"HybridSearch | bm25_rank={bm25_ranks[best_idx]} "
            f"dense_rank={dense_ranks[best_idx]} "
            f"rrf={rrf_scores[best_idx]:.4f} | doc={doc_id[:12]}"
        )
        return doc_id, doc_text

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _load_corpus(self) -> None:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Corpus not found at '{self.data_path}'. "
                "Run python src/data_ingestion.py first."
            )
        with open(self.data_path, "r", encoding="utf-8") as f:
            self._corpus = json.load(f)
        self._doc_ids   = list(self._corpus.keys())
        self._doc_texts = list(self._corpus.values())

    def _build_bm25(self) -> None:
        tokenised = [text.lower().split() for text in self._doc_texts]
        self._bm25 = BM25Okapi(tokenised)

    def _build_faiss(self) -> None:
        """Lazily embed all documents and build a flat inner-product index."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for HybridColdStorage dense retrieval.\n"
                "Install: pip install faiss-cpu"
            )

        logger.info(
            f"HybridColdStorage: building FAISS flat index for "
            f"{len(self._doc_texts)} documents…"
        )
        embedder = EmbeddingEngine()
        # Embed all documents — uses GPU if available
        doc_vecs = np.vstack([
            embedder.embed(text).reshape(1, -1) for text in self._doc_texts
        ]).astype(np.float32)  # shape (N, 384)

        self._doc_vectors = doc_vecs

        # IndexFlatIP = exact inner-product search (cosine if vectors normalised)
        index = faiss.IndexFlatIP(doc_vecs.shape[1])
        index.add(doc_vecs)
        self._faiss_index = index
        logger.info("HybridColdStorage: FAISS index ready.")

    @staticmethod
    def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
        """
        Convert score array to rank array.
        Rank 0 = highest scoring document.
        """
        # argsort gives indices sorted ascending; reverse for descending scores
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))
        return ranks
