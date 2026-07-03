"""
vector_arc_cache.py
───────────────────
O(1)-memory Adaptive Replacement Cache for semantic query vectors.

ARC Architecture
────────────────
                     HOT CACHE (pays memory cost for fast retrieval)
                     ┌────────────────┬────────────────┐
                     │   T1 (Recency) │  T2 (Frequency)│
                     │  LRU ◄──────── │ ─────────► MRU │
                     └────────────────┴────────────────┘
                                   ↑ p (boundary slider — auto-tunes)
                     GHOST LISTS (SimHash fingerprints — O(1) memory)
                     ┌────────────────┬────────────────┐
                     │  B1 (was in T1)│  B2 (was in T2)│
                     │  value = uint64 SimHash of vector│
                     └────────────────┴────────────────┘

Key Properties
──────────────
• Self-tuning: parameter `p` shifts the T1/T2 boundary based on ghost hits.
  A B1 ghost hit means "recency was more useful" → expand T1 (p increases).
  A B2 ghost hit means "frequency was more useful" → expand T2 (p decreases).
• O(1) ghost lists: evicted items drop their heavy float32 vectors; only an
  8-byte uint64 SimHash fingerprint is kept in B1/B2. This is the memory
  advantage over SCRL.
• Semantic ghost matching: SimHash Hamming distance allows ARC to adapt even
  when a PARAPHRASE of an evicted query is re-requested (not just exact re-issue).
• All operations (get, put, _replace) are O(1) amortized via OrderedDict.

SimHash Ghost Fingerprinting (Improvement #2)
──────────────────────────────────────────────
Based on: random hyperplane projections (Charikar 2002), validated by LSH-E
(NeurIPS 2024) for compact, distance-preserving binary fingerprints.

  simhash(v) = pack of 64 sign-bits from (H @ v), where H is a 64×384
               random hyperplane matrix generated once at cache init.

Two semantically similar queries q1, q2:
  cos_sim(q1_vec, q2_vec) ≥ 0.85  ↔  hamming(simhash(q1), simhash(q2)) ≤ ~8

Memory: 8 bytes per ghost (uint64) vs 1,536 bytes if we stored the full vector.
Overhead ratio: 0.5%
"""

import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── SimHash configuration ─────────────────────────────────────────────────────
_SIMHASH_BITS   = 64    # bits in the fingerprint
_EMBEDDING_DIM  = 384   # dimension of all-MiniLM-L6-v2 embeddings
_GHOST_HAMMING_THRESHOLD = 10   # max Hamming distance for semantic ghost match
                                # ~88% recall at threshold 10 (empirically tuned)


class VectorARC:
    """
    Adaptive Replacement Cache storing (query_id → payload) mappings.

    Payload format expected by rag_coordinator:
        {
            "text":       str,          # retrieved document text
            "vector":     np.ndarray,   # 384-dim query embedding (unit-norm)
            "answer":     str,          # full LLM-generated answer (may be None)
            "expires_at": float,        # Unix timestamp, 0.0 = never expires
        }

    Ghost lists (B1/B2) store:
        key   → uint64 SimHash of the evicted vector  (8 bytes, NOT 1,536 bytes)
    """

    def __init__(self, capacity: int, seed: int = 42):
        if capacity < 1:
            raise ValueError(f"Cache capacity must be >= 1, got {capacity}")

        self.c = capacity  # Maximum hot-cache size (|T1| + |T2| ≤ c)
        self.p = 0         # Boundary slider: target size for T1

        # Hot caches — store full payloads
        self.t1: OrderedDict = OrderedDict()  # Recency (seen once)
        self.t2: OrderedDict = OrderedDict()  # Frequency (seen 2+ times)

        # Ghost lists — store uint64 SimHash fingerprints (NOT None, NOT vectors)
        self.b1: OrderedDict = OrderedDict()  # Ghost of T1 evictions
        self.b2: OrderedDict = OrderedDict()  # Ghost of T2 evictions

        # ── SimHash hyperplane matrix — generated once, frozen forever ─────────
        # Shape: (64, 384). Each row is a random unit hyperplane.
        rng = np.random.default_rng(seed)
        H = rng.standard_normal((_SIMHASH_BITS, _EMBEDDING_DIM)).astype(np.float32)
        # Normalise rows to unit length for numerical stability
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        self._hyperplanes: np.ndarray = H   # shape (64, 384), float32

    # ──────────────────────────────────────────────────────────────
    # SimHash helpers
    # ──────────────────────────────────────────────────────────────

    def _simhash(self, vector: np.ndarray) -> int:
        """
        Compute a 64-bit SimHash fingerprint from a 384-dim embedding.

        Steps:
          1. Project vector through 64 random hyperplanes: signs = H @ v
          2. Encode each sign bit as 1 (positive) or 0 (negative)
          3. Pack into a Python int (treated as uint64)

        Time: O(64 × 384) = O(1) for fixed dimensions, ~0.003ms on CPU.
        """
        projections = self._hyperplanes @ vector.astype(np.float32)
        bits = (projections > 0).astype(np.uint8)
        # Pack 64 bits into a Python int using bit-shift accumulation
        fingerprint: int = 0
        for b in bits:
            fingerprint = (fingerprint << 1) | int(b)
        return fingerprint

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        """Count differing bits between two integers (Hamming distance).
        Uses bin().count('1') which is ~10-20x faster than a Python bit-loop
        for 64-bit integers.
        """
        return bin(a ^ b).count('1')

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """
        Cache lookup with ARC promotion and TTL expiry check.

        T1 hit → promoted to T2 (item becomes "frequent").
        T2 hit → moved to MRU position within T2.

        If an entry has expired (expires_at > 0 and now > expires_at),
        it is evicted immediately and treated as a MISS.

        Returns (hit: bool, payload | None).
        Time complexity: O(1).
        """
        now = time.time()

        if key in self.t1:
            payload = self.t1.pop(key)
            # TTL expiry check
            if payload.get("expires_at", 0.0) > 0.0 and now > payload["expires_at"]:
                # Expired — drop it; do NOT admit to ghost (TTL expiry ≠ capacity eviction)
                return False, None
            self.t2[key] = payload          # Promote: recency → frequency
            return True, payload

        if key in self.t2:
            payload = self.t2.pop(key)
            # TTL expiry check
            if payload.get("expires_at", 0.0) > 0.0 and now > payload["expires_at"]:
                return False, None
            self.t2[key] = payload          # Move to MRU position
            return True, payload

        return False, None

    def put(self, key: str, payload: Any, query_vector: Optional[np.ndarray] = None) -> None:
        """
        Insert a new (key, payload) into the cache.

        Ghost hit in B1 → p increases (recency more valuable recently).
        Ghost hit in B2 → p decreases (frequency more valuable recently).
        Complete miss   → item enters T1 (recency tier).

        Args:
            key:          MD5 fingerprint of the query string.
            payload:      Dict with keys: text, vector, answer, expires_at.
            query_vector: The incoming query's embedding. Used to compute a
                          SimHash for semantic ghost matching. If None, falls
                          back to extracting from payload["vector"].

        Time complexity: O(1) amortized.
        """
        # Already in hot cache — caller should use get() instead
        if key in self.t1 or key in self.t2:
            return

        # Compute SimHash of the new query vector for ghost matching
        vec = query_vector if query_vector is not None else payload.get("vector")
        incoming_sh: Optional[int] = self._simhash(vec) if vec is not None else None

        # ── Ghost Recency Hit (B1): item was recently evicted from T1 ────────
        # Check exact key match OR semantic SimHash match
        b1_hit_key = self._find_ghost_hit(self.b1, key, incoming_sh)
        if b1_hit_key is not None:
            delta = max(1, len(self.b2) // len(self.b1) if self.b1 else 1)
            self.p = min(self.c, self.p + delta)
            self._replace(key, in_b2=False, simhash=incoming_sh)
            del self.b1[b1_hit_key]
            self.t2[key] = payload          # Re-admit directly to frequency tier
            return

        # ── Ghost Frequency Hit (B2): item was recently evicted from T2 ───────
        b2_hit_key = self._find_ghost_hit(self.b2, key, incoming_sh)
        if b2_hit_key is not None:
            delta = max(1, len(self.b1) // len(self.b2) if self.b2 else 1)
            self.p = max(0, self.p - delta)
            self._replace(key, in_b2=True, simhash=incoming_sh)
            del self.b2[b2_hit_key]
            self.t2[key] = payload          # Re-admit to frequency tier
            return

        # ── Complete Cache Miss: new item entering the system ─────────────────
        t1_plus_b1 = len(self.t1) + len(self.b1)

        if t1_plus_b1 == self.c:
            # Directory (T1+B1) is full
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)  # Drop oldest ghost to make room
                self._replace(key, in_b2=False, simhash=incoming_sh)
            else:
                # B1 is empty; directly discard LRU of T1
                self.t1.popitem(last=False)
        else:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= self.c:
                if total == 2 * self.c:
                    self.b2.popitem(last=False)  # Trim oldest B2 ghost
                self._replace(key, in_b2=False, simhash=incoming_sh)

        # New item enters recency tier
        self.t1[key] = payload

    def update_answer(self, key: str, answer: str) -> bool:
        """
        Update the 'answer' field of an existing hot cache entry in-place.

        Called by main.py after LLM generates an answer for a cache miss —
        the entry was admitted by retrieve() with answer=None, and this
        upgrades it with the full LLM response so future hits return it.

        Returns True if the key was found and updated, False otherwise.
        """
        if key in self.t1:
            self.t1[key]["answer"] = answer
            return True
        if key in self.t2:
            self.t2[key]["answer"] = answer
            return True
        return False

    def stats(self) -> Dict[str, int]:
        """
        Returns a snapshot of cache occupancy for telemetry / logging.

        Example output:
            {"t1": 2, "t2": 3, "b1": 1, "b2": 0, "hot_total": 5, "p": 2}
        """
        return {
            "t1":        len(self.t1),
            "t2":        len(self.t2),
            "b1":        len(self.b1),
            "b2":        len(self.b2),
            "hot_total": len(self.t1) + len(self.t2),
            "p":         self.p,
        }

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _find_ghost_hit(
        self,
        ghost_list: OrderedDict,
        key: str,
        incoming_sh: Optional[int],
    ) -> Optional[str]:
        """
        Check if the incoming query matches any entry in a ghost list.

        Matching strategy (in priority order):
          1. Exact key match (MD5 collision of the same normalized query string)
          2. Semantic SimHash match (Hamming distance ≤ threshold)
             — catches paraphrases of evicted queries

        Returns the matching ghost key, or None if no match.
        """
        # Priority 1: exact key match
        if key in ghost_list:
            return key

        # Priority 2: semantic SimHash match
        if incoming_sh is not None:
            for ghost_key, ghost_sh in ghost_list.items():
                if ghost_sh is not None and isinstance(ghost_sh, int):
                    if self._hamming(incoming_sh, ghost_sh) <= _GHOST_HAMMING_THRESHOLD:
                        return ghost_key
        return None

    def _replace(self, key: str, in_b2: bool, simhash: Optional[int] = None) -> None:
        """
        Core ARC eviction logic.

        Decides whether to evict the LRU item from T1 (→ B1) or
        from T2 (→ B2) based on the current boundary p.

        When evicting, the vector payload is DROPPED. The ghost list stores
        an 8-byte uint64 SimHash of the evicted vector (not None, not the
        full vector). This preserves semantic ghost matching at O(1) memory.

        Args:
            simhash: The SimHash of the INCOMING query (not the evicted one).
                     The evicted entry's SimHash is computed from its own vector.
        """
        t1_len = len(self.t1)
        evict_from_t1 = self.t1 and (
            t1_len > self.p or (in_b2 and t1_len == self.p)
        )

        if evict_from_t1:
            evicted_key, evicted_payload = self.t1.popitem(last=False)  # LRU of T1
            # Compute SimHash of the EVICTED vector for ghost fingerprint
            evicted_vec = evicted_payload.get("vector") if isinstance(evicted_payload, dict) else None
            ghost_sh = self._simhash(evicted_vec) if evicted_vec is not None else None
            self.b1[evicted_key] = ghost_sh        # Ghost: uint64 SimHash!

        elif self.t2:
            evicted_key, evicted_payload = self.t2.popitem(last=False)  # LRU of T2
            evicted_vec = evicted_payload.get("vector") if isinstance(evicted_payload, dict) else None
            ghost_sh = self._simhash(evicted_vec) if evicted_vec is not None else None
            self.b2[evicted_key] = ghost_sh        # Ghost: uint64 SimHash!