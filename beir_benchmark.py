"""
beir_benchmark.py
─────────────────
Vector-ARC Comprehensive BEIR Multi-Dataset Benchmark

Evaluation Criteria Covered
────────────────────────────
  1. Retrieval Accuracy  : nDCG@10, Hit@1, MRR@10 per dataset
  2. Storage Efficiency  : corpus RAM (MB), index build time, docs/MB ratio
  3. System Design       : BM25-only vs Hybrid BM25+FAISS+RRF vs Hybrid+Reranker
  4. Experimental Analysis: per-dataset breakdown + aggregate summary table
  5. Cost-Efficient RAG  : cache hit rate, P50/P95 latency, LLM calls saved

Datasets Tested (BEIR benchmark — same as SCRL)
────────────────────────────────────────────────
  arguana   — Argument retrieval       (~8K docs)
  fiqa      — Financial QA             (~57K docs)
  hotpotqa  — Multi-hop QA             (sampled ~5K docs)
  msmarco   — Web passages             (sampled ~5K docs)
  nfcorpus  — Biomedical IR            (~3.6K docs)
  quora     — Duplicate questions      (sampled ~5K docs)
  scidocs   — Scientific paper retr.   (~25K docs)
  scifact   — Scientific claims        (~5.2K docs)

All datasets are downloaded on-demand via HuggingFace BeIR.
A --max-docs flag caps corpus size for fast testing.

Usage
─────
  # Fast test: 500 docs, 50 queries, no reranker
  python beir_benchmark.py --max-docs 500 --max-queries 50 --no-reranker

  # Small datasets only
  python beir_benchmark.py --datasets scifact nfcorpus --max-queries 200

  # Full BEIR sweep
  python beir_benchmark.py --all-datasets --max-queries 300

Output
──────
  metrics/beir_results.csv           Full per-metric results
  metrics/beir_summary.txt           Formatted tables
  metrics/beir_storage_report.csv    Storage efficiency
  metrics/beir_cache_simulation.csv  Cache cost analysis
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

METRICS_DIR = "metrics"
DATA_DIR    = "data"

# ── BEIR dataset registry ─────────────────────────────────────────────────────
BEIR_REGISTRY = {
    "arguana":  ("BeIR/arguana",  "BeIR/arguana-qrels",  "test"),
    "fiqa":     ("BeIR/fiqa",     "BeIR/fiqa-qrels",     "test"),
    "hotpotqa": ("BeIR/hotpotqa", "BeIR/hotpotqa-qrels", "test"),
    "msmarco":  ("BeIR/msmarco",  "BeIR/msmarco-qrels",  "validation"),
    "nfcorpus": ("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", "test"),
    "quora":    ("BeIR/quora",    "BeIR/quora-qrels",    "test"),
    "scidocs":  ("BeIR/scidocs",  "BeIR/scidocs-qrels",  "test"),
    "scifact":  ("BeIR/scifact",  "BeIR/scifact-qrels",  "test"),
}

DEFAULT_DATASETS = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "quora"]
ALL_DATASETS     = list(BEIR_REGISTRY.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetMetrics:
    dataset:         str
    pipeline:        str
    n_docs:          int   = 0
    n_queries:       int   = 0
    hit_at_1:        float = 0.0
    ndcg_at_10:      float = 0.0
    mrr_at_10:       float = 0.0
    p50_ms:          float = 0.0
    p95_ms:          float = 0.0
    index_build_s:   float = 0.0
    corpus_ram_mb:   float = 0.0
    cache_hit_rate:  float = 0.0
    llm_calls_saved: int   = 0


@dataclass
class StorageReport:
    dataset:       str
    n_docs:        int
    corpus_ram_mb: float
    docs_per_mb:   float
    bm25_build_s:  float
    faiss_build_s: float
    total_build_s: float
    index_ram_mb:  float


# ─────────────────────────────────────────────────────────────────────────────
# BEIR data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_beir_dataset(
    dataset_name: str,
    max_docs: int = 5000,
    max_queries: int = 300,
) -> Tuple[Dict[str, str], List[Dict], Dict[str, Set[str]]]:
    """
    Download a BEIR dataset via HuggingFace.
    Returns: corpus {doc_id: text}, queries [{qid, text}], qrels {qid: set(doc_ids)}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    if dataset_name not in BEIR_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    hf_corpus, hf_qrels, qrel_split = BEIR_REGISTRY[dataset_name]

    # ── Load corpus ────────────────────────────────────────────────────────────
    logger.info(f"[{dataset_name}] Loading corpus from {hf_corpus}...")
    corpus_ds = load_dataset(hf_corpus, "corpus", split="corpus")

    corpus: Dict[str, str] = {}
    for row in corpus_ds:
        doc_id = str(row["_id"])
        title  = (row.get("title") or "").strip()
        text   = (row.get("text")  or "").strip()
        full   = f"{title}. {text}".strip(". ") if title else text
        if full:
            corpus[doc_id] = full
        if len(corpus) >= max_docs:
            break

    logger.info(f"[{dataset_name}] Corpus: {len(corpus)} docs (cap={max_docs})")

    # ── Load qrels ─────────────────────────────────────────────────────────────
    logger.info(f"[{dataset_name}] Loading qrels...")
    qrels_ds = load_dataset(hf_qrels, split=qrel_split)

    qrels: Dict[str, Set[str]] = {}
    for row in qrels_ds:
        qid    = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        score  = int(row.get("score", 1))
        if score > 0 and doc_id in corpus:
            qrels.setdefault(qid, set()).add(doc_id)

    # ── Load queries ───────────────────────────────────────────────────────────
    logger.info(f"[{dataset_name}] Loading queries...")
    query_ds = load_dataset(hf_corpus, "queries", split="queries")

    queries = []
    for row in query_ds:
        qid = str(row["_id"])
        if qid in qrels:
            queries.append({"qid": qid, "text": row["text"]})
    queries = queries[:max_queries]

    logger.info(
        f"[{dataset_name}] {len(queries)} evaluable queries "
        f"| {sum(len(v) for v in qrels.values())} relevant pairs in corpus"
    )
    if not queries:
        logger.warning(
            f"[{dataset_name}] WARNING: No evaluable queries found! "
            "All relevant docs may be outside the capped corpus. "
            "Try --max-docs with a larger value."
        )

    return corpus, queries, qrels


def save_corpus_json(corpus: Dict[str, str], dataset_name: str) -> str:
    """Write corpus to data/<dataset>_beir.json and return path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{dataset_name}_beir.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f)
    size_mb = os.path.getsize(path) / 1_048_576
    logger.info(f"[{dataset_name}] Corpus saved -> {path} ({size_mb:.2f} MB)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def hit_at_1(retrieved_id: Optional[str], relevant_ids: Set[str]) -> float:
    if retrieved_id is None:
        return 0.0
    return 1.0 if retrieved_id in relevant_ids else 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    dcg  = sum(1.0 / np.log2(r + 2)
               for r, d in enumerate(retrieved_ids[:k]) if d in relevant_ids)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Storage efficiency measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_storage(corpus: Dict[str, str], dataset_name: str) -> StorageReport:
    """Measures memory footprint and index build times."""
    import sys
    from rank_bm25 import BM25Okapi

    doc_texts = list(corpus.values())

    # Corpus RAM
    corpus_bytes  = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in corpus.items())
    corpus_ram_mb = corpus_bytes / 1_048_576

    # BM25 build
    t0 = time.perf_counter()
    BM25Okapi([t.lower().split() for t in doc_texts])
    bm25_build_s = time.perf_counter() - t0

    # FAISS build
    faiss_build_s = 0.0
    index_ram_mb  = 0.0
    try:
        import faiss
        from src.embedder import EmbeddingEngine
        embedder = EmbeddingEngine()
        t0 = time.perf_counter()
        doc_vecs = embedder.embed(doc_texts).astype(np.float32)
        idx = faiss.IndexFlatIP(doc_vecs.shape[1])
        idx.add(doc_vecs)
        faiss_build_s = time.perf_counter() - t0
        index_ram_mb  = doc_vecs.nbytes / 1_048_576
    except ImportError:
        logger.warning("faiss-cpu not installed — skipping FAISS build measurement.")

    docs_per_mb = len(corpus) / corpus_ram_mb if corpus_ram_mb > 0 else 0.0
    report = StorageReport(
        dataset       = dataset_name,
        n_docs        = len(corpus),
        corpus_ram_mb = round(corpus_ram_mb, 3),
        docs_per_mb   = round(docs_per_mb, 1),
        bm25_build_s  = round(bm25_build_s,  3),
        faiss_build_s = round(faiss_build_s, 3),
        total_build_s = round(bm25_build_s + faiss_build_s, 3),
        index_ram_mb  = round(index_ram_mb,  3),
    )
    logger.info(
        f"[{dataset_name}] Storage | corpus={corpus_ram_mb:.2f}MB "
        f"index={index_ram_mb:.2f}MB "
        f"BM25={bm25_build_s:.2f}s FAISS={faiss_build_s:.2f}s"
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval pipelines
# ─────────────────────────────────────────────────────────────────────────────

def run_bm25_pipeline(
    corpus_path: str,
    queries: List[Dict],
    qrels: Dict[str, Set[str]],
    dataset_name: str,
) -> Tuple[pd.DataFrame, List[float]]:
    from src.cold_storage import BM25ColdStorage
    logger.info(f"[{dataset_name}] BM25-only baseline...")
    storage   = BM25ColdStorage(data_path=corpus_path)
    records, latencies = [], []
    for q in queries:
        t0     = time.perf_counter()
        result = storage.search(q["text"])
        lat    = (time.perf_counter() - t0) * 1000
        rid    = result[0] if result else None
        rel    = qrels.get(q["qid"], set())
        records.append({
            "qid": q["qid"], "query": q["text"][:80],
            "retrieved": rid,
            "hit@1":   hit_at_1(rid, rel),
            "ndcg@10": ndcg_at_k([rid] if rid else [], rel),
            "mrr@10":  mrr_at_k([rid]  if rid else [], rel),
            "latency_ms": round(lat, 3),
        })
        latencies.append(lat)
    return pd.DataFrame(records), latencies


def run_hybrid_pipeline(
    corpus_path: str,
    queries: List[Dict],
    qrels: Dict[str, Set[str]],
    dataset_name: str,
    use_reranker: bool = False,
) -> Tuple[pd.DataFrame, List[float]]:
    from src.hybrid_cold_storage import HybridColdStorage
    from src.embedder import EmbeddingEngine
    label = "Hybrid+Reranker" if use_reranker else "Hybrid BM25+FAISS+RRF"
    logger.info(f"[{dataset_name}] {label}...")
    embedder = EmbeddingEngine()
    storage  = HybridColdStorage(
        data_path=corpus_path, embedder=embedder,
        use_reranker=use_reranker, rerank_top_k=5,
    )
    records, latencies = [], []
    for q in queries:
        t0     = time.perf_counter()
        result = storage.search(q["text"])
        lat    = (time.perf_counter() - t0) * 1000
        rid    = result[0] if result else None
        rel    = qrels.get(q["qid"], set())
        records.append({
            "qid": q["qid"], "query": q["text"][:80],
            "retrieved": rid,
            "hit@1":   hit_at_1(rid, rel),
            "ndcg@10": ndcg_at_k([rid] if rid else [], rel),
            "mrr@10":  mrr_at_k([rid]  if rid else [], rel),
            "latency_ms": round(lat, 3),
        })
        latencies.append(lat)
    return pd.DataFrame(records), latencies


# ─────────────────────────────────────────────────────────────────────────────
# Vector-ARC cache simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_cache_simulation(
    corpus_path: str,
    queries: List[Dict],
    dataset_name: str,
    cache_capacity: int = 50,
    similarity_threshold: float = 0.90,
) -> Dict:
    """
    Simulate Vector-ARC caching on a query stream.
    First occurrence of each query = cold miss (cold storage hit).
    Semantically similar subsequent queries = cache hit (LLM call saved).
    """
    from src.rag_coordinator import AdaptiveRAGSystem
    from src.cold_storage import BM25ColdStorage

    logger.info(
        f"[{dataset_name}] Cache simulation "
        f"(capacity={cache_capacity}, threshold={similarity_threshold})..."
    )
    cold = BM25ColdStorage(data_path=corpus_path)
    rag  = AdaptiveRAGSystem(
        cache_capacity=cache_capacity,
        similarity_threshold=similarity_threshold,
        cold_storage=cold,
        ttl_seconds=0,
    )

    hits, misses         = 0, 0
    latencies_hit        = []
    latencies_miss       = []

    for q in queries:
        t0                    = time.perf_counter()
        # retrieve() returns (answer_or_context, is_hit, latency_ms)
        # On a miss, the coordinator already calls admit_to_cache() internally.
        answer, is_hit, _lat  = rag.retrieve(q["text"])
        lat                   = (time.perf_counter() - t0) * 1000

        if is_hit:
            hits += 1
            latencies_hit.append(lat)
        else:
            misses += 1
            latencies_miss.append(lat)

    total    = hits + misses
    hit_rate = hits / total if total > 0 else 0.0
    result   = {
        "dataset":         dataset_name,
        "total_queries":   total,
        "cache_hits":      hits,
        "cache_misses":    misses,
        "hit_rate":        round(hit_rate, 4),
        "llm_calls_saved": hits,
        "p50_hit_ms":      round(np.percentile(latencies_hit,  50) if latencies_hit  else 0, 2),
        "p95_hit_ms":      round(np.percentile(latencies_hit,  95) if latencies_hit  else 0, 2),
        "p50_miss_ms":     round(np.percentile(latencies_miss, 50) if latencies_miss else 0, 2),
        "p95_miss_ms":     round(np.percentile(latencies_miss, 95) if latencies_miss else 0, 2),
        "cache_capacity":  cache_capacity,
        "threshold":       similarity_threshold,
    }
    logger.info(
        f"[{dataset_name}] Cache | hit_rate={hit_rate*100:.1f}% "
        f"({hits}/{total}) | LLM saved: {hits}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_dataset(
    dataset_name: str,
    max_docs: int,
    max_queries: int,
    use_reranker: bool,
    run_cache_sim: bool,
    cache_capacity: int,
) -> Tuple[List[DatasetMetrics], Optional[StorageReport], Optional[Dict]]:

    logger.info(f"\n{'='*70}")
    logger.info(f"  DATASET: {dataset_name.upper()}")
    logger.info(f"{'='*70}")

    try:
        corpus, queries, qrels = load_beir_dataset(
            dataset_name, max_docs=max_docs, max_queries=max_queries
        )
    except Exception as e:
        logger.error(f"[{dataset_name}] Load failed: {e}")
        return [], None, None

    if not queries:
        logger.warning(f"[{dataset_name}] Skipped — no evaluable queries.")
        return [], None, None

    corpus_path = save_corpus_json(corpus, dataset_name)
    all_metrics: List[DatasetMetrics] = []

    # Storage efficiency
    storage_report = None
    try:
        storage_report = measure_storage(corpus, dataset_name)
    except Exception as e:
        logger.warning(f"[{dataset_name}] Storage measurement error: {e}")

    # Pipeline 1: BM25
    try:
        df, lats = run_bm25_pipeline(corpus_path, queries, qrels, dataset_name)
        all_metrics.append(DatasetMetrics(
            dataset=dataset_name, pipeline="BM25 (baseline)",
            n_docs=len(corpus), n_queries=len(queries),
            hit_at_1=round(df["hit@1"].mean(), 4),
            ndcg_at_10=round(df["ndcg@10"].mean(), 4),
            mrr_at_10=round(df["mrr@10"].mean(), 4),
            p50_ms=round(np.percentile(lats, 50), 2),
            p95_ms=round(np.percentile(lats, 95), 2),
            index_build_s=storage_report.bm25_build_s if storage_report else 0,
            corpus_ram_mb=storage_report.corpus_ram_mb if storage_report else 0,
        ))
        df.to_csv(os.path.join(METRICS_DIR, f"beir_{dataset_name}_bm25.csv"), index=False)
    except Exception as e:
        logger.error(f"[{dataset_name}] BM25 failed: {e}\n{traceback.format_exc()}")

    # Pipeline 2: Hybrid BM25+FAISS+RRF
    try:
        df, lats = run_hybrid_pipeline(corpus_path, queries, qrels, dataset_name, use_reranker=False)
        all_metrics.append(DatasetMetrics(
            dataset=dataset_name, pipeline="Hybrid BM25+FAISS+RRF",
            n_docs=len(corpus), n_queries=len(queries),
            hit_at_1=round(df["hit@1"].mean(), 4),
            ndcg_at_10=round(df["ndcg@10"].mean(), 4),
            mrr_at_10=round(df["mrr@10"].mean(), 4),
            p50_ms=round(np.percentile(lats, 50), 2),
            p95_ms=round(np.percentile(lats, 95), 2),
            index_build_s=storage_report.faiss_build_s if storage_report else 0,
            corpus_ram_mb=storage_report.index_ram_mb  if storage_report else 0,
        ))
        df.to_csv(os.path.join(METRICS_DIR, f"beir_{dataset_name}_hybrid.csv"), index=False)
    except Exception as e:
        logger.error(f"[{dataset_name}] Hybrid failed: {e}\n{traceback.format_exc()}")

    # Pipeline 3: Hybrid + Reranker (optional)
    if use_reranker:
        try:
            df, lats = run_hybrid_pipeline(corpus_path, queries, qrels, dataset_name, use_reranker=True)
            all_metrics.append(DatasetMetrics(
                dataset=dataset_name, pipeline="Hybrid+Reranker",
                n_docs=len(corpus), n_queries=len(queries),
                hit_at_1=round(df["hit@1"].mean(), 4),
                ndcg_at_10=round(df["ndcg@10"].mean(), 4),
                mrr_at_10=round(df["mrr@10"].mean(), 4),
                p50_ms=round(np.percentile(lats, 50), 2),
                p95_ms=round(np.percentile(lats, 95), 2),
            ))
            df.to_csv(os.path.join(METRICS_DIR, f"beir_{dataset_name}_reranker.csv"), index=False)
        except Exception as e:
            logger.error(f"[{dataset_name}] Reranker failed: {e}\n{traceback.format_exc()}")

    # Cache simulation
    cache_result = None
    if run_cache_sim:
        try:
            cache_result = run_cache_simulation(
                corpus_path, queries, dataset_name,
                cache_capacity=cache_capacity,
            )
            if all_metrics:
                m = all_metrics[-1]
                m.cache_hit_rate  = cache_result["hit_rate"]
                m.llm_calls_saved = cache_result["llm_calls_saved"]
        except Exception as e:
            logger.error(f"[{dataset_name}] Cache sim failed: {e}\n{traceback.format_exc()}")

    return all_metrics, storage_report, cache_result


# ─────────────────────────────────────────────────────────────────────────────
# Report printers
# ─────────────────────────────────────────────────────────────────────────────

def print_retrieval_table(all_metrics: List[DatasetMetrics]) -> str:
    sep   = "=" * 102
    lines = [
        "", sep,
        "  VECTOR-ARC  x  BEIR  --  RETRIEVAL QUALITY BENCHMARK",
        sep,
        f"  {'Dataset':<12} {'Pipeline':<26} {'Docs':>7} {'Queries':>8} "
        f"{'Hit@1':>7} {'nDCG@10':>8} {'MRR@10':>8} {'P50ms':>7} {'P95ms':>7}",
        "  " + "-" * 98,
    ]
    cur_ds = None
    for m in sorted(all_metrics, key=lambda x: (x.dataset, x.pipeline)):
        ds    = m.dataset if m.dataset != cur_ds else ""
        cur_ds = m.dataset
        lines.append(
            f"  {ds:<12} {m.pipeline:<26} {m.n_docs:>7,} {m.n_queries:>8} "
            f"{m.hit_at_1:>7.3f} {m.ndcg_at_10:>8.3f} {m.mrr_at_10:>8.3f} "
            f"{m.p50_ms:>7.1f} {m.p95_ms:>7.1f}"
        )
    hybrid = [m for m in all_metrics if "Hybrid" in m.pipeline and "Reranker" not in m.pipeline]
    if hybrid:
        lines.append("  " + "-" * 98)
        lines.append(
            f"  {'AVERAGE':<12} {'Hybrid BM25+FAISS+RRF':<26} {'':>7} {'':>8} "
            f"{np.mean([m.hit_at_1   for m in hybrid]):>7.3f} "
            f"{np.mean([m.ndcg_at_10 for m in hybrid]):>8.3f} "
            f"{np.mean([m.mrr_at_10  for m in hybrid]):>8.3f} "
            f"{np.mean([m.p50_ms     for m in hybrid]):>7.1f} "
            f"{np.mean([m.p95_ms     for m in hybrid]):>7.1f}"
        )
    lines += [sep, ""]
    out = "\n".join(lines)
    print(out)
    return out


def print_storage_table(reports: List[StorageReport]) -> str:
    if not reports:
        return ""
    sep   = "=" * 90
    lines = [
        "", sep,
        "  VECTOR-ARC  --  STORAGE EFFICIENCY REPORT",
        sep,
        f"  {'Dataset':<12} {'Docs':>7} {'CorpusMB':>9} {'Docs/MB':>8} "
        f"{'BM25 build':>11} {'FAISS build':>12} {'IndexMB':>8}",
        "  " + "-" * 86,
    ]
    for r in reports:
        lines.append(
            f"  {r.dataset:<12} {r.n_docs:>7,} {r.corpus_ram_mb:>9.2f} "
            f"{r.docs_per_mb:>8.1f} "
            f"{r.bm25_build_s:>9.2f}s  {r.faiss_build_s:>10.2f}s  "
            f"{r.index_ram_mb:>8.2f}"
        )
    lines += [sep, ""]
    out = "\n".join(lines)
    print(out)
    return out


def print_cache_table(cache_results: List[Dict]) -> str:
    if not cache_results:
        return ""
    sep   = "=" * 85
    lines = [
        "", sep,
        "  VECTOR-ARC  --  COST-EFFICIENT RAG  (Cache Simulation)",
        sep,
        f"  {'Dataset':<12} {'Queries':>8} {'Hits':>6} {'HitRate':>8} "
        f"{'LLMSaved':>9} {'P50hit':>7} {'P50miss':>8} {'P95miss':>8}",
        "  " + "-" * 81,
    ]
    for r in cache_results:
        lines.append(
            f"  {r['dataset']:<12} {r['total_queries']:>8} {r['cache_hits']:>6} "
            f"{r['hit_rate']*100:>7.1f}% "
            f"{r['llm_calls_saved']:>9} "
            f"{r['p50_hit_ms']:>6.1f}ms "
            f"{r['p50_miss_ms']:>7.1f}ms "
            f"{r['p95_miss_ms']:>7.1f}ms"
        )
    total_q   = sum(r["total_queries"]   for r in cache_results)
    total_hit = sum(r["cache_hits"]      for r in cache_results)
    total_sav = sum(r["llm_calls_saved"] for r in cache_results)
    lines.append("  " + "-" * 81)
    lines.append(
        f"  {'TOTAL':<12} {total_q:>8} {total_hit:>6} "
        f"{total_hit/total_q*100:>7.1f}% {total_sav:>9}"
    )
    lines += [sep, ""]
    out = "\n".join(lines)
    print(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vector-ARC BEIR Multi-Dataset Benchmark"
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help=f"Datasets. Default: {DEFAULT_DATASETS}")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run ALL BEIR datasets")
    parser.add_argument("--max-docs", type=int, default=5000,
                        help="Max corpus docs per dataset (default: 5000)")
    parser.add_argument("--max-queries", type=int, default=150,
                        help="Max queries per dataset (default: 150)")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Disable cross-encoder reranking")
    parser.add_argument("--no-cache-sim", action="store_true",
                        help="Skip cache simulation")
    parser.add_argument("--cache-capacity", type=int, default=50,
                        help="Vector-ARC cache capacity (default: 50)")
    args = parser.parse_args()

    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,    exist_ok=True)

    datasets = ALL_DATASETS if args.all_datasets else args.datasets
    logger.info(f"Benchmarking: {datasets}")
    logger.info(f"max_docs={args.max_docs} max_queries={args.max_queries} "
                f"reranker={not args.no_reranker} cache_sim={not args.no_cache_sim}")

    all_metrics:      List[DatasetMetrics] = []
    all_storage:      List[StorageReport]  = []
    all_cache_results: List[Dict]          = []

    t_start = time.perf_counter()
    for ds in datasets:
        try:
            metrics, storage, cache_r = benchmark_dataset(
                dataset_name  = ds,
                max_docs      = args.max_docs,
                max_queries   = args.max_queries,
                use_reranker  = not args.no_reranker,
                run_cache_sim = not args.no_cache_sim,
                cache_capacity = args.cache_capacity,
            )
            all_metrics.extend(metrics)
            if storage:      all_storage.append(storage)
            if cache_r:      all_cache_results.append(cache_r)
        except Exception as e:
            logger.error(f"[{ds}] FAILED: {e}\n{traceback.format_exc()}")

    elapsed = time.perf_counter() - t_start
    logger.info(f"\nTotal time: {elapsed:.1f}s")

    if not all_metrics:
        logger.error("No results collected.")
        sys.exit(1)

    ret_table   = print_retrieval_table(all_metrics)
    stor_table  = print_storage_table(all_storage)
    cache_table = print_cache_table(all_cache_results)

    # Save summary
    summary_path = os.path.join(METRICS_DIR, "beir_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(ret_table + stor_table + cache_table)
        f.write(f"\nTotal benchmark time: {elapsed:.1f}s\n")
    logger.info(f"Summary -> {summary_path}")

    # Save CSVs
    pd.DataFrame([asdict(m) for m in all_metrics]).to_csv(
        os.path.join(METRICS_DIR, "beir_results.csv"), index=False)
    if all_storage:
        pd.DataFrame([asdict(r) for r in all_storage]).to_csv(
            os.path.join(METRICS_DIR, "beir_storage_report.csv"), index=False)
    if all_cache_results:
        pd.DataFrame(all_cache_results).to_csv(
            os.path.join(METRICS_DIR, "beir_cache_simulation.csv"), index=False)

    logger.info("BEIR benchmark complete.")


if __name__ == "__main__":
    main()
