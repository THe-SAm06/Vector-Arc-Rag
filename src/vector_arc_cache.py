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
                     GHOST LISTS (string IDs only — O(1) memory)
                     ┌────────────────┬────────────────┐
                     │  B1 (was in T1)│  B2 (was in T2)│
                     └────────────────┴────────────────┘

Key Properties
──────────────
• Self-tuning: parameter `p` shifts the T1/T2 boundary based on ghost hits.
  A B1 ghost hit means "recency was more useful" → expand T1 (p increases).
  A B2 ghost hit means "frequency was more useful" → expand T2 (p decreases).
• O(1) ghost lists: evicted items drop their heavy float32 vectors; only
  their string ID is kept in B1/B2. This is the memory advantage over SCRL.
• All operations (get, put, _replace) are O(1) amortized via OrderedDict.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class VectorARC:
    """
    Adaptive Replacement Cache storing (query_id → payload) mappings.

    Payload format expected by rag_coordinator:
        {"text": str, "vector": np.ndarray}

    The cache never stores raw payloads in the ghost lists (B1/B2) —
    only the string key survives eviction, proving O(1) memory efficiency.
    """

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError(f"Cache capacity must be >= 1, got {capacity}")

        self.c = capacity  # Maximum hot-cache size (|T1| + |T2| ≤ c)
        self.p = 0         # Boundary slider: target size for T1

        # Hot caches — store full payloads
        self.t1: OrderedDict = OrderedDict()  # Recency (seen once)
        self.t2: OrderedDict = OrderedDict()  # Frequency (seen 2+ times)

        # Ghost lists — store ONLY string IDs (no vectors, no text)
        self.b1: OrderedDict = OrderedDict()  # Ghost of T1 evictions
        self.b2: OrderedDict = OrderedDict()  # Ghost of T2 evictions

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        """
        Cache lookup with ARC promotion.

        T1 hit → promoted to T2 (item becomes "frequent").
        T2 hit → moved to MRU position within T2.

        Returns (hit: bool, payload | None).
        Time complexity: O(1).
        """
        if key in self.t1:
            payload = self.t1.pop(key)
            self.t2[key] = payload          # Promote: recency → frequency
            return True, payload

        if key in self.t2:
            payload = self.t2.pop(key)
            self.t2[key] = payload          # Move to MRU position
            return True, payload

        return False, None

    def put(self, key: str, payload: Any) -> None:
        """
        Insert a new (key, payload) into the cache.

        Ghost hit in B1 → p increases (recency more valuable recently).
        Ghost hit in B2 → p decreases (frequency more valuable recently).
        Complete miss   → item enters T1 (recency tier).

        Time complexity: O(1) amortized.
        """
        # Already in hot cache — caller should use get() instead
        if key in self.t1 or key in self.t2:
            return

        # ── Ghost Recency Hit (B1): item was recently evicted from T1 ────────
        if key in self.b1:
            delta = max(1, len(self.b2) // len(self.b1) if self.b1 else 1)
            self.p = min(self.c, self.p + delta)
            self._replace(key, in_b2=False)
            self.b1.pop(key)
            self.t2[key] = payload          # Re-admit directly to frequency tier
            return

        # ── Ghost Frequency Hit (B2): item was recently evicted from T2 ───────
        if key in self.b2:
            delta = max(1, len(self.b1) // len(self.b2) if self.b2 else 1)
            self.p = max(0, self.p - delta)
            self._replace(key, in_b2=True)
            self.b2.pop(key)
            self.t2[key] = payload          # Re-admit to frequency tier
            return

        # ── Complete Cache Miss: new item entering the system ─────────────────
        t1_plus_b1 = len(self.t1) + len(self.b1)

        if t1_plus_b1 == self.c:
            # Directory (T1+B1) is full
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)  # Drop oldest ghost to make room
                self._replace(key, in_b2=False)
            else:
                # B1 is empty; directly discard LRU of T1
                self.t1.popitem(last=False)
        else:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= self.c:
                if total == 2 * self.c:
                    self.b2.popitem(last=False)  # Trim oldest B2 ghost
                self._replace(key, in_b2=False)

        # New item enters recency tier
        self.t1[key] = payload

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

    def _replace(self, key: str, in_b2: bool) -> None:
        """
        Core ARC eviction logic.

        Decides whether to evict the LRU item from T1 (→ B1) or
        from T2 (→ B2) based on the current boundary p.

        When evicting to a ghost list, ONLY the string key is stored —
        the vector payload is dropped entirely (O(1) ghost memory).
        """
        t1_len = len(self.t1)
        evict_from_t1 = self.t1 and (
            t1_len > self.p or (in_b2 and t1_len == self.p)
        )

        if evict_from_t1:
            evicted_key, _ = self.t1.popitem(last=False)  # LRU of T1
            self.b1[evicted_key] = None                    # Ghost: key only!
        elif self.t2:
            evicted_key, _ = self.t2.popitem(last=False)  # LRU of T2
            self.b2[evicted_key] = None                    # Ghost: key only!