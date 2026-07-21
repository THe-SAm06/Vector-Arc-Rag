"""
main.py
───────
Interactive end-to-end demo for the Vector-ARC RAG pipeline.

Usage
─────
  # Full pipeline (cache + LLM answer generation):
  python main.py

  # Cache-only mode (no LLM API needed):
  python main.py --no-llm

  # Resume from saved cache state:
  python main.py --load-state

What changed in v2
──────────────────
  • Cache hits now return the stored LLM answer — LLM skipped on hits
  • Hybrid cold storage (BM25 + FAISS + RRF) used by default
  • Cache state saved to disk on exit (--load-state to resume)
  • Margin guard rejects ambiguous matches before serving cached answers
"""

import argparse
import atexit
import logging

from src.rag_coordinator import AdaptiveRAGSystem

logging.basicConfig(level=logging.INFO, format="%(message)s")

_STATE_PATH = "cache_state/session"


def run_demonstration(use_llm: bool = True, load_state: bool = False) -> None:
    print("🚀 Initialising Vector-ARC RAG Pipeline (v2)...\n")

    rag = AdaptiveRAGSystem(
        cache_capacity=5,
        similarity_threshold=0.90,     # raised from 0.85
        data_path="data/scifact_corpus_full.json",
        ttl_seconds=86400,             # 24-hour TTL
    )

    if load_state:
        loaded = rag.load_state(_STATE_PATH)
        if loaded:
            print(f"♻️  Resumed from saved cache state ({_STATE_PATH})\n")

    # Save state on exit automatically
    atexit.register(rag.save_state, _STATE_PATH)

    llm = None
    if use_llm:
        try:
            from src.llm_engine import LLMEngine
            llm = LLMEngine()
        except Exception as e:
            print(f"⚠️  LLM init failed ({e}). Running in cache-only mode.\n")
            use_llm = False

    # Strategic query sequence exercising every ARC state transition.
    # Added semantic paraphrase pairs to demonstrate SimHash ghost matching.
    test_queries = [
        "What is the curfew for first-year students?",     # Miss  → T1
        "What is the curfew for first-year students?",     # Hit   T1 → T2 (LLM skipped)
        "Tell me about the attendance policy.",             # Miss  → T1
        "Tell me about the attendance policy.",             # Hit   T1 → T2 (LLM skipped)
        "What is the curfew for first-year students?",     # Hit   T2 (freq) (LLM skipped)
        "When is the library open?",                       # Miss  → eviction
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] User: {query}")
        print("─" * 50)

        # retrieve() returns raw_text on miss (no LLM answer yet)
        context, cache_hit, latency_ms = rag.retrieve(query)
        status = "✅ HIT" if cache_hit else "❌ MISS"
        print(f"Cache status : {status}  ({latency_ms:.2f}ms)")

        if cache_hit:
            # Cache hit: context IS the cached LLM answer — print directly
            print(f"🤖 Answer    : {context}")
        elif use_llm and llm is not None:
            # Cache miss: retrieve() already admitted to cache with answer=None.
            # Generate LLM answer and upgrade the cache entry in-place.
            answer = llm.generate_answer(query, context)
            if answer.startswith("System Error:"):
                err_hint = "Invalid API key" if "401" in answer else "LLM unavailable"
                print(f"⚠️  {err_hint} — showing retrieved context:")
                print(f"📄 Context   : {context[:300]}{'...' if len(context) > 300 else ''}")
            else:
                print(f"🤖 Answer    : {answer}")
                # Upgrade cache entry with the full LLM answer
                from src.rag_coordinator import _query_fingerprint
                qkey = _query_fingerprint(query)
                rag.cache.update_answer(qkey, answer)
        else:
            # Cache-only mode
            print(f"📄 Retrieved : {context[:300]}{'...' if len(context) > 300 else ''}")

    s = rag.cache.stats()
    print("\n" + "=" * 60)
    print("📊 FINAL PIPELINE METRICS")
    print("=" * 60)
    print(f"  Total queries      : {rag.metrics['total_queries']}")
    print(f"  Cache hits         : {rag.metrics['hits']}")
    print(f"  Cache misses       : {rag.metrics['misses']}")
    print(f"  Hit rate           : {rag.hit_rate()*100:.1f}%")
    print(f"  Margin rejections  : {rag.metrics['margin_rejections']}")
    print(f"  TTL expirations    : {rag.metrics['ttl_expirations']}")
    print(f"  Cold storage calls : {rag.metrics['cold_storage_calls']}")
    print(f"  Boundary P         : {s['p']}")
    print(f"  T1/T2/B1/B2        : {s['t1']}/{s['t2']}/{s['b1']}/{s['b2']}")
    print(
        "\nOn cache hits, LLM is skipped — latency should be ~1ms vs ~300ms on misses."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector-ARC RAG Pipeline Demo v2")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in cache-only mode (skip LLM — useful when API quota is exhausted)",
    )
    parser.add_argument(
        "--load-state",
        action="store_true",
        help="Resume from previously saved cache state",
    )
    args = parser.parse_args()
    run_demonstration(use_llm=not args.no_llm, load_state=args.load_state)