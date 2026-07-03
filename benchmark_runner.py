"""
benchmark_runner.py
───────────────────
Vector-ARC Algorithm Validation Benchmark

Purpose
───────
This script proves that the Vector-ARC caching algorithm behaves correctly
by walking through all six ARC state transitions in a deterministic,
phase-labelled sequence. It is designed to be run live as a professor demo.

Six Phases
──────────
  Phase 1 – Cold Start      : All queries miss (cache is empty)
  Phase 2 – Recency Hits    : Re-issuing queries hits T1, promotes items to T2
  Phase 3 – Frequency Hits  : Re-issuing queries hits T2 (already frequent)
  Phase 4 – Eviction        : New queries force old items from T1 into ghost B1
  Phase 5 – Ghost / Adapt   : Re-issuing evicted queries triggers B1 ghost hits;
                               boundary parameter p shifts upward (ARC adapts)
  Phase 6 – Novel Miss      : Completely new query is handled gracefully

What to observe
───────────────
  • Hit latency  ≈ 1–5ms   (serving from in-memory numpy dot product)
  • Miss latency ≈ 30–1500ms (embedding + BM25 cold storage)
  • Ghost memory stays tiny (only string keys, no vectors)
  • Boundary p: starts at 0, rises to ~2 after ghost hits in Phase 5

Usage
─────
  cd /path/to/vector_arc/
  python benchmark_runner.py
"""

import collections  # Fix: must be imported at module level for calculate_memory_bytes
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd

from src.rag_coordinator import AdaptiveRAGSystem
from src.cold_storage import BM25ColdStorage

# ─────────────────────────────────────────────────────────────────────────────
# Logging — INFO level so we see the coordinator's hit/miss log lines
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CORPUS_PATH = "data/scifact_corpus.json"

# Cache capacity deliberately small to force evictions quickly during the demo.
# In production (main.py) the default is 50.
DEMO_CACHE_CAPACITY = 5

# Q-to-Q similarity threshold — must be ≥ this to register a cache hit.
DEMO_THRESHOLD = 0.90

METRICS_DIR = "metrics"
OUTPUT_CSV = os.path.join(METRICS_DIR, "vector_arc_performance.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Workload builder
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_workload(corpus: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Constructs a 17-step, phase-labelled workload from the SciFact corpus.

    Each entry is a dict:
        {"query": str, "phase": str, "label": str, "expected": "HIT"|"MISS"}

    The workload is engineered so that (with capacity=5):
      • Steps  1-5  fill T1 completely  (5 cold misses)
      • Steps  6-8  promote items T1→T2 (recency hits)
      • Steps  9-11 serve from T2        (frequency hits)
      • Steps 12-14 evict T1 items → B1  (eviction pressure)
      • Steps 15-16 trigger B1 ghost hits, p: 0→1→2 (ARC adaptation)
      • Step  17    novel query; handled gracefully (miss)
    """
    texts = list(corpus.values())

    # Pick 5 well-separated topics from the corpus
    q1 = " ".join(texts[0].split()[:10])    # Topic A
    q2 = " ".join(texts[20].split()[:10])   # Topic B
    q3 = " ".join(texts[50].split()[:10])   # Topic C
    q4 = " ".join(texts[80].split()[:10])   # Topic D (will be evicted in P4)
    q5 = " ".join(texts[120].split()[:10])  # Topic E (will be evicted in P4)

    # 3 new topics that create eviction pressure in Phase 4
    q6 = " ".join(texts[150].split()[:10])  # Topic F
    q7 = " ".join(texts[200].split()[:10])  # Topic G
    q8 = " ".join(texts[250].split()[:10])  # Topic H

    workload = [
        # ── Phase 1: Cold Start ──────────────────────────────────────────────
        {"query": q1, "phase": "1 · Cold Start",   "label": "A", "expected": "MISS"},
        {"query": q2, "phase": "1 · Cold Start",   "label": "B", "expected": "MISS"},
        {"query": q3, "phase": "1 · Cold Start",   "label": "C", "expected": "MISS"},
        {"query": q4, "phase": "1 · Cold Start",   "label": "D", "expected": "MISS"},
        {"query": q5, "phase": "1 · Cold Start",   "label": "E", "expected": "MISS"},
        # ── Phase 2: Recency Hits (T1 → T2 promotion) ───────────────────────
        {"query": q1, "phase": "2 · Recency Hit",  "label": "A", "expected": "HIT"},
        {"query": q2, "phase": "2 · Recency Hit",  "label": "B", "expected": "HIT"},
        {"query": q3, "phase": "2 · Recency Hit",  "label": "C", "expected": "HIT"},
        # ── Phase 3: Frequency Hits (served from T2) ─────────────────────────
        {"query": q1, "phase": "3 · Frequency Hit","label": "A", "expected": "HIT"},
        {"query": q2, "phase": "3 · Frequency Hit","label": "B", "expected": "HIT"},
        {"query": q3, "phase": "3 · Frequency Hit","label": "C", "expected": "HIT"},
        # ── Phase 4: Eviction (D and E get pushed to ghost B1) ───────────────
        {"query": q6, "phase": "4 · Eviction",     "label": "F", "expected": "MISS"},
        {"query": q7, "phase": "4 · Eviction",     "label": "G", "expected": "MISS"},
        {"query": q8, "phase": "4 · Eviction",     "label": "H", "expected": "MISS"},
        # ── Phase 5: Ghost Hits — ARC adapts, p increases ───────────────────
        # Ghost hits (B1) are internal ARC signals: they shift p upward and
        # re-admit the item to T2, but the payload was dropped on eviction so
        # cold storage is still called. From the user's view: MISS.
        # Proof of correctness: watch p go 0 → 1 → 2 across these two steps.
        {"query": q4, "phase": "5 · Ghost / Adapt","label": "D*","expected": "MISS"},
        {"query": q5, "phase": "5 · Ghost / Adapt","label": "E*","expected": "MISS"},
        # ── Phase 6: Novel Miss ───────────────────────────────────────────────
        {
            "query": "quantum entanglement effects in biological neural tissue synapses",
            "phase": "6 · Novel Miss",
            "label": "novel",
            "expected": "MISS",
        },
    ]
    return workload


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def object_size_bytes(obj: Any) -> int:
    """
    Recursively estimates memory footprint of a nested Python object.
    Handles dict, list, tuple, set, deque, and numpy arrays.
    Uses a seen-set to avoid double-counting shared references.
    """
    seen_ids = set()

    def _size(o: Any) -> int:
        if id(o) in seen_ids:
            return 0
        seen_ids.add(id(o))
        s = sys.getsizeof(o)
        if isinstance(o, dict):
            s += sum(_size(k) + _size(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, collections.deque)):
            s += sum(_size(i) for i in o)
        # numpy arrays: sys.getsizeof already returns the buffer size for ndarray
        return s

    return _size(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

class VectorARCBenchmark:
    """Runs the phased demo workload and collects per-step telemetry."""

    def __init__(self, corpus_path: str = CORPUS_PATH):
        self.corpus_path = corpus_path
        self.corpus: Dict[str, str] = self._load_corpus()

    def _load_corpus(self) -> Dict[str, str]:
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(
                f"Corpus not found at '{self.corpus_path}'. "
                "Run python src/data_ingestion.py first."
            )
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        logger.info(f"Loaded {len(corpus):,} documents from '{self.corpus_path}'")
        return corpus

    def run(self) -> pd.DataFrame:
        workload = build_demo_workload(self.corpus)

        _print_banner()
        print(f"  Cache capacity : {DEMO_CACHE_CAPACITY}")
        print(f"  Similarity threshold : {DEMO_THRESHOLD}")
        print(f"  Cold storage   : BM25 ({len(self.corpus):,} docs)")
        print(f"  Total steps    : {len(workload)}")
        print()

        system = AdaptiveRAGSystem(
            cache_capacity=DEMO_CACHE_CAPACITY,
            similarity_threshold=DEMO_THRESHOLD,
            cold_storage=BM25ColdStorage(data_path=CORPUS_PATH),
        )

        records: List[Dict] = []
        current_phase = ""

        for step_idx, item in enumerate(workload, start=1):
            phase    = item["phase"]
            label    = item["label"]
            query    = item["query"]
            expected = item["expected"]

            # ── Print phase header ──────────────────────────────────────────
            if phase != current_phase:
                current_phase = phase
                print(f"\n{'─'*70}")
                print(f"  {phase}")
                print(f"{'─'*70}")

            # ── Execute retrieval ───────────────────────────────────────────
            _, cache_hit, latency_ms = system.retrieve(query)

            # ── Gather cache state for telemetry ───────────────────────────
            s = system.cache.stats()
            ghost_mem  = system.ghost_memory_bytes()
            hot_mem    = system.hot_cache_memory_bytes()
            actual     = "HIT" if cache_hit else "MISS"
            correct    = actual == expected
            status_sym = "✓" if correct else "✗"
            hit_sym    = "✅" if cache_hit else "❌"

            # ── Console output ──────────────────────────────────────────────
            print(
                f"  [{step_idx:2d}] Topic {label:<6} {hit_sym} {actual:<4}  {status_sym}  "
                f"{latency_ms:8.2f}ms  "
                f"T1={s['t1']} T2={s['t2']} B1={s['b1']} B2={s['b2']}  p={s['p']}"
            )

            records.append({
                "Step":             step_idx,
                "Phase":            phase,
                "Topic":            label,
                "Query_Snippet":    query[:60] + "…",
                "Expected":         expected,
                "Actual":           actual,
                "Correct":          correct,
                "Cache_Hit":        cache_hit,
                "Latency_ms":       round(latency_ms, 4),
                "T1_Size":          s["t1"],
                "T2_Size":          s["t2"],
                "B1_Size":          s["b1"],
                "B2_Size":          s["b2"],
                "Hot_Total":        s["hot_total"],
                "Boundary_P":       s["p"],
                "Hot_Memory_Bytes": hot_mem,
                "Ghost_Memory_Bytes": ghost_mem,
            })

        # ── Summary ─────────────────────────────────────────────────────────
        df = pd.DataFrame(records)
        _print_summary(system, df)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  VECTOR-ARC ALGORITHM VALIDATION BENCHMARK".center(68) + "║")
    print("║" + "  IIT Dharwad Summer of Innovation".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()


def _print_summary(system: AdaptiveRAGSystem, df: pd.DataFrame) -> None:
    total   = len(df)
    hits    = df["Cache_Hit"].sum()
    misses  = total - hits
    correct = df["Correct"].sum()

    hit_latencies  = df[df["Cache_Hit"]  == True]["Latency_ms"]
    miss_latencies = df[df["Cache_Hit"] == False]["Latency_ms"]

    avg_hit_ms  = hit_latencies.mean()  if len(hit_latencies)  else 0.0
    avg_miss_ms = miss_latencies.mean() if len(miss_latencies) else 0.0

    # Final cache state
    s = system.cache.stats()
    final_ghost_bytes = system.ghost_memory_bytes()
    final_hot_bytes   = system.hot_cache_memory_bytes()

    # Phase-level hit rate
    phase_summary = (
        df.groupby("Phase")
          .apply(lambda g: f"{g['Cache_Hit'].sum()}/{len(g)} hits", include_groups=False)
          .reset_index()
    )
    phase_summary.columns = ["Phase", "Result"]

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  BENCHMARK RESULTS SUMMARY".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Total Queries':<35} {total:>30} ║")
    print(f"║  {'Cache Hits':<35} {hits:>30} ║")
    print(f"║  {'Cache Misses':<35} {misses:>30} ║")
    print(f"║  {'Hit Rate':<35} {system.hit_rate()*100:>29.1f}% ║")
    print(f"║  {'Steps Behaving as Expected':<35} {correct}/{total:>27} ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Avg Hit  Latency':<35} {avg_hit_ms:>27.2f}ms ║")
    print(f"║  {'Avg Miss Latency':<35} {avg_miss_ms:>27.2f}ms ║")
    print(f"║  {'Speedup (Miss / Hit)':<35} {avg_miss_ms/max(avg_hit_ms,0.01):>28.1f}x ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Final Boundary P':<35} {s['p']:>30} ║")
    print(f"║  {'Hot  Cache (T1+T2) memory':<35} {final_hot_bytes:>26,} B ║")
    print(f"║  {'Ghost Lists (B1+B2) memory':<35} {final_ghost_bytes:>26,} B ║")
    print(f"║  {'T1 / T2 / B1 / B2 sizes':<35} {s['t1']}/{s['t2']}/{s['b1']}/{s['b2']:>24} ║")
    print("╠" + "═" * 68 + "╣")
    print("║  Phase Breakdown:".ljust(69) + "║")
    for _, row in phase_summary.iterrows():
        line = f"    {row['Phase']:<40} {row['Result']}"
        print(f"║  {line:<66} ║")
    print("╚" + "═" * 68 + "╝")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    profiler = VectorARCBenchmark(corpus_path=CORPUS_PATH)
    df = profiler.run()

    os.makedirs(METRICS_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  📄 Telemetry saved → {OUTPUT_CSV}")
    print()