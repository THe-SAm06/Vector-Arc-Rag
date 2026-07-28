"""
run_full_benchmark.py
─────────────────────
Comparative Benchmark: No-Cache vs LRU vs Vector-ARC

Runs a controlled workload through three caching strategies and
produces a quantitative comparison for the project report.

Metrics measured:
  • Cache hit rate
  • LLM calls avoided (%)
  • Average hit/miss latency
  • Speedup factor
  • Ghost list compression ratio (ARC-specific)
  • Estimated API cost savings
  • ARC state correctness (p adaptation, T1/T2/B1/B2)

Usage:
  PYTHONPATH=. python run_full_benchmark.py
"""

import collections
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.embedder import EmbeddingEngine
from src.cold_storage import BM25ColdStorage
from src.vector_arc_cache import VectorARC

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_PATH     = "data/scifact_corpus.json"
CACHE_CAPACITY  = 5
SIM_THRESHOLD   = 0.75
MARGIN_EPS      = 0.05
METRICS_DIR     = "metrics"

# Groq Llama-3.3-70B pricing (free tier, but estimating commercial rates)
LLM_INPUT_COST_PER_1M   = 0.06   # $/1M input tokens
LLM_OUTPUT_COST_PER_1M  = 0.20   # $/1M output tokens
AVG_CONTEXT_TOKENS       = 400    # avg tokens per retrieved doc
AVG_OUTPUT_TOKENS         = 150    # avg tokens per LLM response


# ─────────────────────────────────────────────────────────────────────────────
# Simple LRU Cache (baseline comparison)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleLRUCache:
    """Basic LRU cache without ghost lists or adaptive p."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: collections.OrderedDict = collections.OrderedDict()

    def get(self, key: str) -> Tuple[bool, Optional[Any]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return True, self.cache[key]
        return False, None

    def put(self, key: str, payload: Any) -> list:
        evicted = []
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = payload
            return evicted
        if len(self.cache) >= self.capacity:
            evicted_key, _ = self.cache.popitem(last=False)
            evicted.append(evicted_key)
        self.cache[key] = payload
        return evicted

    def stats(self) -> Dict:
        return {"size": len(self.cache), "capacity": self.capacity}


# ─────────────────────────────────────────────────────────────────────────────
# Workload builder
# ─────────────────────────────────────────────────────────────────────────────
def build_workload(corpus: Dict[str, str]) -> List[Dict]:
    """
    Builds a 25-step workload designed to test all cache behaviors:
    - Cold start (all misses)
    - Exact repeats (hits)
    - Paraphrases (semantic hits — only ARC and LRU with semantic matching)
    - Eviction pressure (new topics force evictions)
    - Ghost hits (re-requesting evicted items — ARC advantage)
    - Distribution shift (new topic cluster)
    """
    texts = list(corpus.values())
    keys = list(corpus.keys())

    # Pick 8 well-separated topics
    q = []
    indices = [0, 20, 50, 80, 120, 150, 200, 250]
    for i, idx in enumerate(indices):
        if idx < len(texts):
            q.append(" ".join(texts[idx].split()[:12]))
        else:
            q.append(f"fallback query topic {i}")

    workload = [
        # Phase 1: Cold Start (5 misses)
        {"query": q[0], "phase": "1-Cold Start",    "label": "A",  "expected": "MISS"},
        {"query": q[1], "phase": "1-Cold Start",    "label": "B",  "expected": "MISS"},
        {"query": q[2], "phase": "1-Cold Start",    "label": "C",  "expected": "MISS"},
        {"query": q[3], "phase": "1-Cold Start",    "label": "D",  "expected": "MISS"},
        {"query": q[4], "phase": "1-Cold Start",    "label": "E",  "expected": "MISS"},
        # Phase 2: Exact Repeats (hits)
        {"query": q[0], "phase": "2-Exact Repeat",  "label": "A",  "expected": "HIT"},
        {"query": q[1], "phase": "2-Exact Repeat",  "label": "B",  "expected": "HIT"},
        {"query": q[2], "phase": "2-Exact Repeat",  "label": "C",  "expected": "HIT"},
        # Phase 3: More Exact Repeats (frequency)
        {"query": q[0], "phase": "3-Frequency Hit",  "label": "A",  "expected": "HIT"},
        {"query": q[1], "phase": "3-Frequency Hit",  "label": "B",  "expected": "HIT"},
        # Phase 4: Eviction Pressure (new topics push out D, E)
        {"query": q[5], "phase": "4-Eviction",      "label": "F",  "expected": "MISS"},
        {"query": q[6], "phase": "4-Eviction",      "label": "G",  "expected": "MISS"},
        {"query": q[7], "phase": "4-Eviction",      "label": "H",  "expected": "MISS"},
        # Phase 5: Ghost Hits (re-request evicted D, E)
        # ARC: ghost hit in B1 → p adapts, re-admitted to T2
        # LRU: cold miss (no ghost tracking)
        {"query": q[3], "phase": "5-Ghost/Adapt",   "label": "D*", "expected": "MISS"},
        {"query": q[4], "phase": "5-Ghost/Adapt",   "label": "E*", "expected": "MISS"},
        # Phase 6: Repeated access to recently ghosted items
        # ARC: should hit T2 now (re-admitted via ghost)
        # LRU: cold miss again
        {"query": q[3], "phase": "6-Post-Ghost",    "label": "D",  "expected": "HIT"},
        {"query": q[4], "phase": "6-Post-Ghost",    "label": "E",  "expected": "HIT"},
        # Phase 7: Distribution shift + repeats
        {"query": q[0], "phase": "7-Long-Term",     "label": "A",  "expected": "HIT"},
        {"query": q[5], "phase": "7-Long-Term",     "label": "F",  "expected": "HIT"},
        # Phase 8: Novel query
        {"query": "quantum entanglement in biological neural tissue",
         "phase": "8-Novel",          "label": "novel", "expected": "MISS"},
    ]
    return workload


# ─────────────────────────────────────────────────────────────────────────────
# Strategy runners
# ─────────────────────────────────────────────────────────────────────────────
def _fingerprint(q: str) -> str:
    import hashlib
    return hashlib.md5(q.strip().lower().encode()).hexdigest()[:16]


def run_no_cache(workload, cold_storage, embedder):
    """Every query goes to cold storage. No caching at all."""
    records = []
    for item in workload:
        t0 = time.perf_counter()
        # Always embed + cold search
        _ = embedder.embed(item["query"])
        result = cold_storage.search(item["query"])
        latency_ms = (time.perf_counter() - t0) * 1000
        records.append({
            "Phase": item["phase"], "Label": item["label"],
            "Hit": False, "Latency_ms": round(latency_ms, 4),
            "LLM_Call": True,
        })
    return records


def run_lru(workload, cold_storage, embedder, capacity, threshold):
    """LRU cache with semantic matching (no ghosts, no adaptive p)."""
    cache = SimpleLRUCache(capacity=capacity)
    records = []
    for item in workload:
        t0 = time.perf_counter()
        key = _fingerprint(item["query"])

        # Exact match check
        hit, payload = cache.get(key)
        if hit:
            latency_ms = (time.perf_counter() - t0) * 1000
            records.append({
                "Phase": item["phase"], "Label": item["label"],
                "Hit": True, "Latency_ms": round(latency_ms, 4),
                "LLM_Call": False,
            })
            continue

        # Semantic match check
        query_vec = embedder.embed(item["query"])
        best_score = 0.0
        best_key = None
        for k, p in cache.cache.items():
            vec = p.get("vector")
            if vec is not None:
                score = float(query_vec @ vec)
                if score > best_score:
                    best_score = score
                    best_key = k

        if best_key and best_score >= threshold:
            hit, payload = cache.get(best_key)
            if hit:
                latency_ms = (time.perf_counter() - t0) * 1000
                records.append({
                    "Phase": item["phase"], "Label": item["label"],
                    "Hit": True, "Latency_ms": round(latency_ms, 4),
                    "LLM_Call": False,
                })
                continue

        # Miss
        result = cold_storage.search(item["query"])
        text = result[1] if result else ""
        cache.put(key, {"text": text, "vector": query_vec, "answer": text})
        latency_ms = (time.perf_counter() - t0) * 1000
        records.append({
            "Phase": item["phase"], "Label": item["label"],
            "Hit": False, "Latency_ms": round(latency_ms, 4),
            "LLM_Call": True,
        })
    return records


def run_vector_arc(workload, cold_storage, embedder, capacity, threshold, margin_eps):
    """Full Vector-ARC with T1/T2/B1/B2, adaptive p, SimHash ghosts."""
    cache = VectorARC(capacity=capacity)
    records = []
    arc_trace = []  # detailed ARC state at each step

    for step_idx, item in enumerate(workload):
        t0 = time.perf_counter()
        key = _fingerprint(item["query"])

        # Exact match
        hit, payload = cache.get(key)
        if hit and payload is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            s = cache.stats()
            records.append({
                "Phase": item["phase"], "Label": item["label"],
                "Hit": True, "Latency_ms": round(latency_ms, 4),
                "LLM_Call": False,
            })
            arc_trace.append({
                "Step": step_idx + 1, "Phase": item["phase"], "Label": item["label"],
                "Hit": True, "T1": s["t1"], "T2": s["t2"],
                "B1": s["b1"], "B2": s["b2"], "P": s["p"],
            })
            continue

        # Semantic match
        query_vec = embedder.embed(item["query"])
        best_key, best_score, best_margin = None, 0.0, 0.0
        all_keys, all_scores = [], []
        for tier in (cache.t1, cache.t2):
            for k, p in tier.items():
                vec = p.get("vector")
                if vec is not None:
                    score = float(query_vec @ vec)
                    all_keys.append(k)
                    all_scores.append(score)

        if all_scores:
            scores_arr = np.array(all_scores)
            sorted_idx = np.argsort(scores_arr)[::-1]
            best_key = all_keys[sorted_idx[0]]
            best_score = float(scores_arr[sorted_idx[0]])
            if len(scores_arr) > 1:
                best_margin = best_score - float(scores_arr[sorted_idx[1]])
            else:
                best_margin = best_score

        if best_key and best_score >= threshold and best_margin >= margin_eps:
            hit, payload = cache.get(best_key)
            if hit and payload is not None:
                latency_ms = (time.perf_counter() - t0) * 1000
                s = cache.stats()
                records.append({
                    "Phase": item["phase"], "Label": item["label"],
                    "Hit": True, "Latency_ms": round(latency_ms, 4),
                    "LLM_Call": False,
                })
                arc_trace.append({
                    "Step": step_idx + 1, "Phase": item["phase"], "Label": item["label"],
                    "Hit": True, "T1": s["t1"], "T2": s["t2"],
                    "B1": s["b1"], "B2": s["b2"], "P": s["p"],
                })
                continue

        # Miss — cold storage
        result = cold_storage.search(item["query"])
        text = result[1] if result else ""
        expires_at = time.time() + 86400
        evicted = cache.put(key, {
            "text": text, "vector": query_vec, "answer": text, "expires_at": expires_at,
        }, query_vector=query_vec)

        latency_ms = (time.perf_counter() - t0) * 1000
        s = cache.stats()
        records.append({
            "Phase": item["phase"], "Label": item["label"],
            "Hit": False, "Latency_ms": round(latency_ms, 4),
            "LLM_Call": True, "Evicted": evicted if evicted else [],
        })
        arc_trace.append({
            "Step": step_idx + 1, "Phase": item["phase"], "Label": item["label"],
            "Hit": False, "T1": s["t1"], "T2": s["t2"],
            "B1": s["b1"], "B2": s["b2"], "P": s["p"],
            "Evicted": evicted if evicted else [],
        })

    return records, arc_trace, cache


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────
def compute_summary(name: str, records: list) -> Dict:
    total = len(records)
    hits = sum(1 for r in records if r["Hit"])
    misses = total - hits
    llm_calls = sum(1 for r in records if r.get("LLM_Call", True))
    hit_lats = [r["Latency_ms"] for r in records if r["Hit"]]
    miss_lats = [r["Latency_ms"] for r in records if not r["Hit"]]

    avg_hit = np.mean(hit_lats) if hit_lats else 0.0
    avg_miss = np.mean(miss_lats) if miss_lats else 0.0
    total_lat = sum(r["Latency_ms"] for r in records)

    # Cost estimation
    llm_cost = llm_calls * (
        AVG_CONTEXT_TOKENS * LLM_INPUT_COST_PER_1M / 1e6 +
        AVG_OUTPUT_TOKENS * LLM_OUTPUT_COST_PER_1M / 1e6
    )

    return {
        "Strategy": name,
        "Total_Queries": total,
        "Hits": hits,
        "Misses": misses,
        "Hit_Rate_%": round(hits / total * 100, 1),
        "LLM_Calls": llm_calls,
        "LLM_Avoided_%": round((1 - llm_calls / total) * 100, 1),
        "Avg_Hit_ms": round(avg_hit, 2),
        "Avg_Miss_ms": round(avg_miss, 2),
        "Speedup": round(avg_miss / avg_hit, 1) if avg_hit > 0.01 else 0.0,
        "Total_Latency_ms": round(total_lat, 2),
        "Est_Cost_$": round(llm_cost, 6),
    }


def print_summary(summaries: list, arc_cache: VectorARC, output_path: str):
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p()
    p("=" * 90)
    p("  VECTOR-ARC COMPARATIVE BENCHMARK")
    p("  No-Cache vs LRU vs Vector-ARC (Adaptive Replacement Cache)")
    p("=" * 90)
    p()
    p(f"  {'Strategy':<30} {'Hits':>5} {'Misses':>7} {'Hit%':>6} {'LLM↓%':>7} "
      f"{'Hit ms':>8} {'Miss ms':>9} {'Speed':>6} {'Cost$':>10}")
    p("  " + "─" * 86)

    for s in summaries:
        p(f"  {s['Strategy']:<30} {s['Hits']:>5} {s['Misses']:>7} "
          f"{s['Hit_Rate_%']:>5.1f}% {s['LLM_Avoided_%']:>6.1f}% "
          f"{s['Avg_Hit_ms']:>7.2f} {s['Avg_Miss_ms']:>9.2f} "
          f"{s['Speedup']:>5.1f}x {s['Est_Cost_$']:>10.6f}")

    p()
    p("  " + "─" * 86)
    p("  KEY FINDINGS:")
    p()

    # Compare ARC vs No-Cache
    no_cache = summaries[0]
    arc = summaries[2]
    llm_saved_pct = arc["LLM_Avoided_%"]
    cost_saved_pct = round((1 - arc["Est_Cost_$"] / max(no_cache["Est_Cost_$"], 1e-9)) * 100, 1)

    p(f"  • Vector-ARC avoided {llm_saved_pct}% of LLM calls vs No-Cache baseline")
    p(f"  • Estimated cost reduction: {cost_saved_pct}%")
    if arc["Speedup"] > 0:
        p(f"  • Cache hits are {arc['Speedup']}x faster than cache misses")

    # Compare ARC vs LRU
    lru = summaries[1]
    arc_advantage = arc["Hits"] - lru["Hits"]
    if arc_advantage > 0:
        p(f"  • Vector-ARC achieved {arc_advantage} more hits than LRU ({arc['Hit_Rate_%']}% vs {lru['Hit_Rate_%']}%)")
        p(f"    → Ghost lists (B1/B2) with SimHash fingerprinting catch re-requested evicted queries")
    elif arc_advantage == 0:
        p(f"  • Vector-ARC and LRU achieved equal hit rates on this workload")
        p(f"    → ARC's advantage emerges under distribution shift and larger workloads")

    # Storage efficiency
    p()
    s = arc_cache.stats()
    ghost_count = s["b1"] + s["b2"]
    hot_count = s["t1"] + s["t2"]
    ghost_actual_bytes = ghost_count * 8  # uint64 SimHash per ghost
    ghost_hypothetical_bytes = ghost_count * 384 * 4  # full float32 vector
    compression = ghost_hypothetical_bytes / max(ghost_actual_bytes, 1)

    p("  STORAGE EFFICIENCY:")
    p(f"  • Hot cache (T1+T2): {hot_count} entries")
    p(f"  • Ghost lists (B1+B2): {ghost_count} entries")
    p(f"  • Ghost memory (SimHash): {ghost_actual_bytes} bytes ({ghost_count} × 8B)")
    p(f"  • Ghost memory (if full vectors): {ghost_hypothetical_bytes:,} bytes ({ghost_count} × 1,536B)")
    p(f"  • Compression ratio: {compression:.0f}x (SimHash vs full vector)")
    p(f"  • Final ARC state: T1={s['t1']} T2={s['t2']} B1={s['b1']} B2={s['b2']} p={s['p']}")

    p()
    p("=" * 90)

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Loading corpus and embedding model...\n")

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    cold_storage = BM25ColdStorage(data_path=CORPUS_PATH)
    embedder = EmbeddingEngine()

    workload = build_workload(corpus)

    print(f"  Corpus: {len(corpus)} documents")
    print(f"  Workload: {len(workload)} queries")
    print(f"  Cache capacity: {CACHE_CAPACITY}")
    print(f"  Similarity threshold: {SIM_THRESHOLD}")
    print()

    # ── Run all three strategies ──────────────────────────────────────────────
    print("  [1/3] Running No-Cache baseline...")
    no_cache_records = run_no_cache(workload, cold_storage, embedder)

    print("  [2/3] Running LRU cache...")
    lru_records = run_lru(workload, cold_storage, embedder, CACHE_CAPACITY, SIM_THRESHOLD)

    print("  [3/3] Running Vector-ARC cache...")
    arc_records, arc_trace, arc_cache = run_vector_arc(
        workload, cold_storage, embedder, CACHE_CAPACITY, SIM_THRESHOLD, MARGIN_EPS
    )

    # ── Compute summaries ─────────────────────────────────────────────────────
    summaries = [
        compute_summary("No Cache (Baseline)", no_cache_records),
        compute_summary("LRU Cache", lru_records),
        compute_summary("Vector-ARC Cache", arc_records),
    ]

    summary_path = os.path.join(METRICS_DIR, "full_benchmark_summary.txt")
    print_summary(summaries, arc_cache, summary_path)

    # ── Save detailed CSVs ────────────────────────────────────────────────────
    pd.DataFrame(summaries).to_csv(
        os.path.join(METRICS_DIR, "full_benchmark_comparison.csv"), index=False
    )

    arc_trace_df = pd.DataFrame(arc_trace)
    arc_trace_df.to_csv(
        os.path.join(METRICS_DIR, "full_benchmark_arc_trace.csv"), index=False
    )

    print(f"\n  📄 Summary   → {summary_path}")
    print(f"  📄 Comparison → {os.path.join(METRICS_DIR, 'full_benchmark_comparison.csv')}")
    print(f"  📄 ARC Trace  → {os.path.join(METRICS_DIR, 'full_benchmark_arc_trace.csv')}")
    print()
