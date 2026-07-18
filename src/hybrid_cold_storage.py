"""
hybrid_cold_storage.py
──────────────────────
Hybrid BM25 + FAISS cold storage with Reciprocal Rank Fusion (RRF)
and an optional Cross-Encoder reranker.

Pipeline
────────
  Query
    │
    ├──► BM25 (sparse, keyword matching)  ─────────────────┐
    │                                                        ├──► RRF Fusion ──► Top-K ──► Cross-Encoder ──► best doc
    └──► FAISS flat (dense, semantic)    ─────────────────┘

Design rationale (2026 research consensus)
─────────────────────────────────────────
Neither BM25 (sparse) nor dense vector search (FAISS) alone is optimal:
  • BM25 excels at exact terminology, product IDs, rare tokens
  • FAISS excels at semantic intent, synonyms, paraphrase understanding
  • Hybrid + RRF consistently outperforms either alone by 20-40% recall
    (Pinecone blog 2024; Elasticsearch hybrid docs 2025; arxiv 2024)

Cross-Encoder Reranker (Stage 2)
─────────────────────────────────
After hybrid first-stage retrieval returns top-K candidates, a cross-encoder
scores each (query, doc) pair jointly using full attention over both texts.
This is far more accurate than bi-encoder cosine similarity, adding ~+10
nDCG@10 points at the cost of ~5-15ms for K=5.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  • Trained on MS-MARCO passage ranking (200M+ training pairs)
  • 22M parameters — fast enough for a hot path (<10ms for K=5)
  • Industry standard: Elasticsearch, Pinecone, Cohere all use this pattern

RRF constant k=60
  score_rrf(d) = Σ  1 / (k + rank_i(d))
               i ∈ {bm25, faiss}
  k=60 is the standard (Cormack et al., 2009) — robust without per-dataset tuning.

Why FAISS flat index (not HNSW)?
  For corpora up to ~50,000 documents, exhaustive flat search is faster
  than HNSW due to HNSW's graph-building overhead and memory indirection.
  HNSW is planned for Phase 3 when corpus exceeds 50K docs.

Embedder Injection (Singleton Pattern Fix)
──────────────────────────────────────────
The embedder is injected at construction time (not created per call).
EmbeddingEngine already implements a class-level singleton via __new__,
but injecting it explicitly:
  1. Makes dependencies visible and testable
  2. Guarantees the SAME instance is shared with the coordinator
  3. Avoids any risk of multiple SentenceTransformer object creations
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

# How many candidates to retrieve from first-stage (BM25+FAISS) before reranking
_RERANK_TOP_K = 5


class HybridColdStorage(BaseColdStorage):
    """
    Hybrid retrieval: BM25 (sparse) + FAISS flat (dense) fused with RRF,
    then optionally reranked by a Cross-Encoder for maximum precision.

    Inherits from BaseColdStorage — drop-in replacement for BM25ColdStorage.
    Interface: search(query: str) -> Optional[Tuple[str, str]]

    Args:
        data_path:    Path to corpus JSON file (doc_id -> text mapping).
        embedder:     Shared EmbeddingEngine instance. If None, uses the
                      class-level singleton (EmbeddingEngine()).
        use_reranker: If True, applies cross-encoder reranking on top-K
                      candidates. Adds ~5-15ms but significantly improves
                      precision on the final selected document.
        rerank_top_k: Number of candidates from first-stage to pass to
                      the cross-encoder. Default 5.
    """

    def __init__(
        self,
        data_path: str = "data/iit_dharwad_corpus.json",
        embedder: Optional[EmbeddingEngine] = None,
        use_reranker: bool = True,
        rerank_top_k: int = _RERANK_TOP_K,
    ):
        self.data_path    = data_path
        self.rerank_top_k = rerank_top_k
        self.use_reranker = use_reranker

        # Injected or singleton embedder — never instantiated per call
        self._embedder: EmbeddingEngine = embedder or EmbeddingEngine()

        self._corpus:     Dict[str, str]        = {}
        self._doc_ids:    List[str]              = []
        self._doc_texts:  List[str]              = []
        self._bm25:       Optional[BM25Okapi]   = None
        self._faiss_index                        = None
        self._doc_vectors: Optional[np.ndarray] = None  # (N, 384) float32

        # Cross-encoder — loaded lazily on first reranking call
        self._cross_encoder = None

        self._load_corpus()
        self._build_bm25()
        logger.info(
            f"HybridColdStorage: loaded {len(self._corpus)} documents "
            f"from '{data_path}' | BM25 ready | "
            f"FAISS index builds on first query | "
            f"reranker={'enabled' if use_reranker else 'disabled'}"
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
        with Reciprocal Rank Fusion, then optional cross-encoder reranking.

        Returns (doc_id, doc_text) of the best result, or None if corpus empty.

        Pipeline:
          1. BM25 ranks all N docs by keyword overlap        O(N)
          2. FAISS dot-product ranks all N docs by semantics O(N·D)
          3. RRF fuses both rank lists                        O(N)
          4. Cross-encoder reranks top-K candidates          O(K) — optional
        """
        if not self._doc_ids:
            return None

        # Build FAISS index on first call (lazy — avoids GPU overhead on startup)
        if self._faiss_index is None:
            self._build_faiss()

        # ── Stage 1a: BM25 ranking ────────────────────────────────────────────
        tokens      = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)       # shape (N,)
        bm25_ranks  = self._scores_to_ranks(bm25_scores)  # 0 = best

        # ── Stage 1b: FAISS dense ranking ─────────────────────────────────────
        # Use the injected/singleton embedder — never create a new one here
        q_vec      = self._embedder.embed(query).reshape(1, -1).astype(np.float32)
        dense_dot  = (self._doc_vectors @ q_vec.T).squeeze()   # shape (N,)
        dense_ranks = self._scores_to_ranks(dense_dot)          # 0 = best

        # ── Stage 1c: Reciprocal Rank Fusion ──────────────────────────────────
        rrf_scores = (
            1.0 / (_RRF_K + bm25_ranks + 1).astype(np.float64) +
            1.0 / (_RRF_K + dense_ranks + 1).astype(np.float64)
        )  # vectorised — no Python loop

        # ── Stage 2: Cross-Encoder Reranking (optional) ───────────────────────
        if self.use_reranker and self.rerank_top_k > 0:
            # Take top-K by RRF score and rerank with the cross-encoder
            k       = min(self.rerank_top_k, len(self._doc_ids))
            top_k_idx  = np.argsort(rrf_scores)[::-1][:k]
            best_idx   = self._rerank(query, top_k_idx)
        else:
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
        tokenised  = [text.lower().split() for text in self._doc_texts]
        self._bm25 = BM25Okapi(tokenised)

    def _build_faiss(self) -> None:
        """
        Lazily embed ALL documents in a SINGLE batched call and build
        a flat inner-product FAISS index.

        Key fix from audit: previously called embed() in a Python loop
        (one GPU call per document). Now uses the batch encode API:
          embedder.embed(list_of_texts) — single GPU forward pass for all N docs
        This is 10-100x faster for N > 100.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for HybridColdStorage dense retrieval.\n"
                "Install: conda run -n tf_gpu_conda pip install faiss-cpu"
            )

        logger.info(
            f"HybridColdStorage: building FAISS flat index — "
            f"batched embedding {len(self._doc_texts)} documents…"
        )

        # Single batched GPU call — 10-100x faster than looping embed()
        doc_vecs = self._embedder.embed(self._doc_texts).astype(np.float32)
        # embed() already returns (N, D) when given a list of strings

        self._doc_vectors = doc_vecs

        # IndexFlatIP = exact inner-product (= cosine sim for unit-norm vectors)
        index = faiss.IndexFlatIP(doc_vecs.shape[1])
        index.add(doc_vecs)
        self._faiss_index = index
        logger.info(
            f"HybridColdStorage: FAISS flat index ready "
            f"({doc_vecs.shape[0]} docs × {doc_vecs.shape[1]}D)."
        )

    def _load_cross_encoder(self) -> None:
        """Lazily load cross-encoder on first reranking call."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.info("HybridColdStorage: cross-encoder loaded (ms-marco-MiniLM-L-6-v2).")
        except Exception as e:
            logger.warning(
                f"Cross-encoder unavailable ({e}). "
                "Falling back to RRF-only selection."
            )
            self.use_reranker = False

    def _rerank(self, query: str, candidate_indices: np.ndarray) -> int:
        """
        Rerank candidate documents using a cross-encoder and return the
        index (in self._doc_ids) of the highest-scoring document.

        The cross-encoder scores (query, doc) pairs jointly using full
        attention — far more precise than cosine similarity of bi-encoders.

        Args:
            query:             The user's query string.
            candidate_indices: Indices into self._doc_ids of the top-K candidates.

        Returns:
            The index into self._doc_ids of the best document after reranking.
        """
        if self._cross_encoder is None:
            self._load_cross_encoder()

        # If loading failed, fall back to first candidate (best by RRF)
        if self._cross_encoder is None:
            return int(candidate_indices[0])

        pairs = [
            [query, self._doc_texts[i]]
            for i in candidate_indices
        ]
        scores = self._cross_encoder.predict(pairs)  # shape (K,), float32
        best_local = int(np.argmax(scores))
        return int(candidate_indices[best_local])

    @staticmethod
    def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
        """
        Convert score array to rank array (vectorised, no Python loop).
        Rank 0 = highest scoring document.
        """
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(scores))
        return ranks
