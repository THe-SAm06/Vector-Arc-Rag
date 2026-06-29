"""
main.py
───────
Interactive end-to-end demo for the Vector-ARC RAG pipeline.

Usage
─────
  # Full pipeline (cache + LLM answer generation):
  python main.py

  # Cache-only mode (no LLM API needed — great for demos when quota is low):
  python main.py --no-llm

The cache algorithm is the research contribution. --no-llm still
demonstrates every cache state transition with real latency numbers.
"""

import argparse
import logging

from src.rag_coordinator import AdaptiveRAGSystem

logging.basicConfig(level=logging.INFO, format="%(message)s")


def run_demonstration(use_llm: bool = True) -> None:
    print("🚀 Initialising Vector-ARC RAG Pipeline...\n")

    # capacity=5 keeps evictions visible during a live demo.
    rag = AdaptiveRAGSystem(
        cache_capacity=5,
        similarity_threshold=0.85,
        data_path="data/iit_dharwad_corpus.json",   # IIT Dharwad domain corpus
    )

    llm = None
    if use_llm:
        try:
            from src.llm_engine import LLMEngine
            llm = LLMEngine()
        except Exception as e:
            print(f"⚠️  LLM init failed ({e}). Running in cache-only mode.\n")
            use_llm = False

    # A strategic query sequence to exercise every ARC state transition.
    test_queries = [
        "What is the curfew for first-year students?",   # Miss  → T1
        "What is the curfew for first-year students?",   # Hit   T1 → T2
        "Tell me about the attendance policy.",           # Miss  → T1
        "Tell me about the attendance policy.",           # Hit   T1 → T2
        "What is the curfew for first-year students?",   # Hit   T2 (freq)
        "When is the library open?",                     # Miss  → eviction
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] User: {query}")
        print("─" * 50)

        context, cache_hit, latency_ms = rag.retrieve(query)
        status = "✅ HIT" if cache_hit else "❌ MISS"
        print(f"Cache status : {status}  ({latency_ms:.2f}ms)")

        if use_llm and llm is not None:
            answer = llm.generate_answer(query, context)
            if answer.startswith("System Error:"):
                err_hint = "Invalid API key" if "401" in answer else "LLM unavailable"
                print(f"⚠️  {err_hint} — showing retrieved context:")
                print(f"📄 Context   : {context[:300]}{'...' if len(context) > 300 else ''}")
            else:
                print(f"🤖 Answer    : {answer}")
        else:
            # Cache-only mode: show the retrieved context directly
            print(f"📄 Retrieved : {context[:300]}{'...' if len(context) > 300 else ''}")

    s = rag.cache.stats()
    print("\n" + "=" * 60)
    print("📊 FINAL PIPELINE METRICS")
    print("=" * 60)
    print(f"  Total queries   : {rag.metrics['total_queries']}")
    print(f"  Cache hits      : {rag.metrics['hits']}")
    print(f"  Cache misses    : {rag.metrics['misses']}")
    print(f"  Hit rate        : {rag.hit_rate()*100:.1f}%")
    print(f"  Boundary P      : {s['p']}")
    print(f"  T1/T2/B1/B2     : {s['t1']}/{s['t2']}/{s['b1']}/{s['b2']}")
    print(
        "\nIf hits resolved in ≈1ms while misses took ≈30-1500ms, "
        "Vector-ARC is fully operational."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector-ARC RAG Pipeline Demo")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run in cache-only mode (skip LLM generation — useful when API quota is exhausted)",
    )
    args = parser.parse_args()
    run_demonstration(use_llm=not args.no_llm)