"""
rag_coordinator.py
──────────────────
Master orchestrator for the Vector-ARC RAG pipeline.

Pipeline Flow
─────────────
  User Query
      │
      ▼
  [Embedder] embed(query) → query_vector
      │
      ▼
  [VectorARC Cache] compare query_vector against cached query vectors (Q-to-Q)
      │
      ├─── HIT (cosine_similarity ≥ threshold) ──────────────► return cached text
      │                                                          (microseconds)
      └─── MISS ──────► [Cold Storage] search(query) ──────────► (doc_id, text)
                              │
                              ▼
                         Cache admission:
                         key   = hash(query)     ← query fingerprint
                         value = {text: ...,
                                  vector: query_vector}  ← store QUERY vector!
                              │
                              ▼
                         return text

The Q-to-Q Caching Insight
───────────────────────────
Previously the cache stored the *document* vector against the doc_id key.
This created a cross-space comparison (query vector vs document vector),
yielding low cosine similarity scores (≈ 0.30–0.50) even for semantically
equivalent queries.

Fix: store the *query* vector. Now, when a semantically similar query
arrives, it is compared against a query that was embedded in the same
space — cosine similarity jumps to 0.85+, giving us the high hit rates
we need.

Modularity
──────────
cold_storage is injected via the constructor, so you can swap backends:
  AdaptiveRAGSystem(cold_storage=SQLColdStorage(...))
  AdaptiveRAGSystem(cold_storage=VectorDBColdStorage(...))
without touching this file.
"""

import hashlib
import logging
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np

from src.base_cold_storage import BaseColdStorage
from src.cold_storage import BM25ColdStorage
from src.embedder import EmbeddingEngine
from src.vector_arc_cache import VectorARC

logger = logging.getLogger(__name__)


class AdaptiveRAGSystem:
    """
    Coordinates the Vector-ARC cache, cold storage backend, and query embedder.

    Args:
        cache_capacity:      Maximum number of query vectors in hot cache.
                             Production default: 50. Use smaller (e.g. 5) to
                             force evictions in demos / algorithm validation.
        similarity_threshold: Minimum cosine similarity for a cache hit.
                              0.85 is a strong default for Q-to-Q matching.
        cold_storage:        Injectable cold storage backend.
                             Defaults to BM25ColdStorage (BM25 keyword search).
                             Swap to any BaseColdStorage subclass to experiment.
        data_path:           Path to corpus JSON (used only when cold_storage
                             is not provided explicitly).
    """

    def __init__(
        self,
        cache_capacity: int = 50,
        similarity_threshold: float = 0.85,
        cold_storage: Optional[BaseColdStorage] = None,
        data_path: str = "data/scifact_corpus.json",
    ):
        self.cache = VectorARC(capacity=cache_capacity)
        self.cold_storage: BaseColdStorage = cold_storage or BM25ColdStorage(
            data_path=data_path
        )
        self.embedder = EmbeddingEngine()
        self.threshold = similarity_threshold

        # Telemetry — inspected by benchmark_runner
        self.metrics: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "total_queries": 0,
            "cold_storage_calls": 0,
        }

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def retrieve(self, user_query: str) -> Tuple[str, bool, float]:
        """
        Primary retrieval entry point.

        Args:
            user_query: Raw natural-language query from the user.

        Returns:
            (context_text, cache_hit, latency_ms)
              context_text — the retrieved document text
              cache_hit    — True if served from cache, False if from cold storage
              latency_ms   — wall-clock time for this retrieval
        """
        self.metrics["total_queries"] += 1
        t0 = time.perf_counter()

        # ── Step 1: Embed the query ───────────────────────────────────────────
        query_vector = self.embedder.embed(user_query)

        # ── Step 2: Search the hot cache (Query-to-Query cosine similarity) ──
        best_key, best_score = self._find_best_match(query_vector)

        # ── Step 3: Cache Hit ─────────────────────────────────────────────────
        if best_key is not None and best_score >= self.threshold:
            self.metrics["hits"] += 1
            _, payload = self.cache.get(best_key)
            latency_ms = (time.perf_counter() - t0) * 1_000
            logger.info(
                f"✅ CACHE HIT  | score={best_score:.4f} | {latency_ms:.2f}ms"
            )
            return payload["text"], True, latency_ms

        # ── Step 4: Cache Miss → Cold Storage ────────────────────────────────
        self.metrics["misses"] += 1
        self.metrics["cold_storage_calls"] += 1

        search_result = self.cold_storage.search(user_query)

        if search_result is None:
            latency_ms = (time.perf_counter() - t0) * 1_000
            logger.warning("⚠️  No result found in cold storage.")
            return "No relevant information found in the knowledge base.", False, latency_ms

        _, raw_text = search_result

        # ── Step 5: Admit to cache — store QUERY vector (Q-to-Q pivot) ───────
        query_key = _query_fingerprint(user_query)
        payload = {"text": raw_text, "vector": query_vector}
        self.cache.put(query_key, payload)

        latency_ms = (time.perf_counter() - t0) * 1_000
        logger.info(
            f"❌ CACHE MISS | cold_storage fetch | {latency_ms:.2f}ms"
        )
        return raw_text, False, latency_ms

    def hit_rate(self) -> float:
        """Returns the cumulative cache hit rate as a fraction [0, 1]."""
        total = self.metrics["total_queries"]
        return self.metrics["hits"] / total if total > 0 else 0.0

    def ghost_memory_bytes(self) -> int:
        """
        Measures the memory used by the ghost lists (B1 + B2).
        These store only string keys — demonstrating O(1) overhead.
        """
        b1_bytes = sum(sys.getsizeof(k) for k in self.cache.b1.keys())
        b2_bytes = sum(sys.getsizeof(k) for k in self.cache.b2.keys())
        return b1_bytes + b2_bytes

    def hot_cache_memory_bytes(self) -> int:
        """
        Measures the memory used by the hot cache (T1 + T2).
        These store the full text + query vector payloads.
        """
        def _payload_size(payload: dict) -> int:
            text_bytes = sys.getsizeof(payload.get("text", ""))
            vec = payload.get("vector")
            vec_bytes = vec.nbytes if vec is not None else 0
            return text_bytes + vec_bytes

        t1_bytes = sum(_payload_size(p) for p in self.cache.t1.values())
        t2_bytes = sum(_payload_size(p) for p in self.cache.t2.values())
        return t1_bytes + t2_bytes

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _find_best_match(
        self, query_vector: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Scans T1 and T2 to find the stored query vector with the highest
        cosine similarity to the incoming query.

        Since embeddings are pre-normalised (unit vectors), cosine similarity
        reduces to a dot product — fast and numerically stable.

        Returns (best_key, best_score) or (None, 0.0) if the cache is empty.
        """
        best_key: Optional[str] = None
        best_score: float = 0.0

        for tier in (self.cache.t1, self.cache.t2):
            if not tier:
                continue
            keys = list(tier.keys())
            vectors = np.array([tier[k]["vector"] for k in keys])
            scores = vectors @ query_vector          # dot product = cosine sim

            idx = int(np.argmax(scores))
            if scores[idx] > best_score:
                best_score = float(scores[idx])
                best_key = keys[idx]

        return best_key, best_score


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _query_fingerprint(query: str) -> str:
    """
    Creates a stable, short string key from a query.

    We hash the normalised query so that two differently-cased or
    differently-spaced versions of the same question map to the same key.
    The 16-character hex prefix is collision-resistant enough for cache sizes
    up to tens of thousands of entries.
    """
    normalised = query.strip().lower()
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()[:16]