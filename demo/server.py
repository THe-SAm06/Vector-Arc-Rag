#!/usr/bin/env python3
"""
Vector-ARC RAG Demo Server
──────────────────────────
Exposes the real embedding + ARC cache + LLM pipeline over HTTP.

Usage:
    cd /path/to/vector_arc
    python demo/server.py

Then open: http://localhost:8080
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, time, threading, hashlib, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ─── Global state ─────────────────────────────────────────────────────────────
rag    = None
llm    = None
_lock  = threading.Lock()
_ready = threading.Event()
_init_error = None
_query_log  = []   # last 100 queries for session history

CORPUS_PATH = "data/scifact_corpus_full.json"
CACHE_CAP   = 10
SIM_THRESH  = 0.75


# ─── Initialization (runs in background thread) ───────────────────────────────
def _initialize():
    global rag, llm, _init_error
    try:
        from dotenv import load_dotenv
        load_dotenv()

        from src.duckdb_cold_storage import DuckDBColdStorage
        from src.rag_coordinator import AdaptiveRAGSystem
        print("  Loading embedding model and corpus (20–30 s) …", flush=True)
        t0 = time.perf_counter()
        rag = AdaptiveRAGSystem(
            cache_capacity   = CACHE_CAP,
            similarity_threshold = SIM_THRESH,
            cold_storage     = DuckDBColdStorage(),
            ttl_seconds      = 86_400,
        )
        print(f"  ✅ RAG system ready ({time.perf_counter()-t0:.1f} s)", flush=True)

        try:
            from src.llm_engine import LLMEngine
            llm = LLMEngine()
            print(f"  ✅ LLM ready: {llm.provider} / {llm.model}", flush=True)
        except Exception as e:
            print(f"  ⚠️  LLM unavailable: {e}", flush=True)

        _ready.set()
    except Exception as e:
        _init_error = str(e)
        traceback.print_exc()
        _ready.set()   # unblock status checks even on error


# ─── Per-step timed query execution ──────────────────────────────────────────
def _process_query(query: str) -> dict:
    """
    Runs the full pipeline with per-step timing.
    Does NOT call rag.retrieve() — calls each step directly
    so we can capture fine-grained timing and step logs.
    """
    from src.rag_coordinator import _query_fingerprint

    result = {
        "query":           query,
        "answer":          "",
        "cache_hit":       False,
        "cache_tier":      None,       # "exact" | "T1" | "T2" | "semantic" | None
        "similarity":      None,
        "margin":          None,
        "threshold":       SIM_THRESH,
        "context_preview": "",
        "retrieved_docs":  0,
        "times": {"embed_ms": 0, "cache_ms": 0, "cold_ms": 0, "llm_ms": 0, "total_ms": 0},
        "arc_state":   {},
        "metrics":     {},
        "steps":       [],
    }
    steps = result["steps"]

    def step(msg: str):
        ts = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
        steps.append({"ts": ts, "msg": msg})

    t_wall = time.perf_counter()

    # ── 1. Exact fingerprint lookup (no embedding needed) ──────────────────────
    step("Query received — running exact fingerprint lookup")
    query_key = _query_fingerprint(query)
    t0 = time.perf_counter()
    hit_ok, payload = rag.cache.get(query_key)
    cache_fast_ms = (time.perf_counter() - t0) * 1_000

    if hit_ok and payload is not None:
        result["times"]["cache_ms"] = round(cache_fast_ms, 2)
        result["times"]["embed_ms"] = 0
        result["cache_hit"]  = True
        result["similarity"] = 1.0
        result["margin"]     = 1.0
        result["cache_tier"] = _which_tier(query_key)
        result["answer"]     = payload.get("answer") or payload.get("text", "")
        rag.metrics["hits"]          += 1
        rag.metrics["total_queries"] += 1
        step(f"Exact cache hit — key={query_key[:8]}… — skipping embedding entirely")
        step(f"Served from {result['cache_tier']} — LLM call skipped")
        step(f"Done — {round((time.perf_counter()-t_wall)*1000,1)} ms")
        result["times"]["total_ms"] = round((time.perf_counter()-t_wall)*1000, 2)
        result["arc_state"] = _arc_snapshot()
        result["metrics"]   = _metrics_snapshot()
        _query_log.append({"ts": time.strftime("%H:%M:%S"), "q": query[:55], "hit": True,
                            "ms": result["times"]["total_ms"]})
        if len(_query_log) > 100: _query_log.pop(0)
        return result

    # ── 2. Generate embedding ─────────────────────────────────────────────────
    step("Generating 384-dim query embedding (all-MiniLM-L6-v2)")
    t0 = time.perf_counter()
    query_vector = rag.embedder.embed(query)
    embed_ms = (time.perf_counter() - t0) * 1_000
    result["times"]["embed_ms"] = round(embed_ms, 2)
    step(f"Embedding done — {round(embed_ms,1)} ms, dim=384, unit-normalised")

    # ── 3. Semantic cache scan (cosine similarity over T1 ∪ T2) ───────────────
    hot_count = len(rag.cache.t1) + len(rag.cache.t2)
    step(f"Semantic cache scan — {hot_count} hot entries (T1={len(rag.cache.t1)}, T2={len(rag.cache.t2)})")
    t0 = time.perf_counter()
    best_key, best_score, margin = rag._find_best_match(query_vector)
    cache_ms = (time.perf_counter() - t0) * 1_000
    result["times"]["cache_ms"] = round(cache_ms, 2)
    result["similarity"] = round(best_score, 4) if best_score is not None else None
    result["margin"]     = round(margin, 4) if margin is not None else None

    sim_str    = f"{best_score:.4f}" if best_score is not None else "—"
    margin_str = f"{margin:.4f}" if margin else "—"
    step(f"Best similarity: {sim_str} | margin: {margin_str} | threshold: {SIM_THRESH}")

    # ── Cache hit decision ─────────────────────────────────────────────────────
    is_semantic_hit = (
        best_key  is not None and
        best_score is not None and
        best_score >= rag.threshold and
        margin  is not None and
        margin  >= rag.margin_eps
    )
    is_margin_reject = (
        best_key  is not None and
        best_score is not None and
        best_score >= rag.threshold and
        (margin is None or margin < rag.margin_eps)
    )

    if is_semantic_hit:
        hit_ok2, payload2 = rag.cache.get(best_key)
        if hit_ok2 and payload2 is not None:
            result["cache_hit"]  = True
            result["cache_tier"] = _which_tier(best_key)
            result["answer"]     = payload2.get("answer") or payload2.get("text", "")
            rag.metrics["hits"]          += 1
            rag.metrics["total_queries"] += 1
            step(f"Cache HIT — {best_score:.4f} ≥ {SIM_THRESH} | margin {margin:.4f} ≥ {rag.margin_eps}")
            step(f"Served from {result['cache_tier']} — LLM call skipped")
        else:
            is_semantic_hit = False
            rag.metrics["ttl_expirations"] = rag.metrics.get("ttl_expirations", 0) + 1
            step("Cache entry expired (TTL) — treating as miss")

    if is_margin_reject and not is_semantic_hit:
        rag.metrics["margin_rejections"] = rag.metrics.get("margin_rejections", 0) + 1
        step(f"Margin reject — ambiguous match (margin {margin:.4f} < {rag.margin_eps}) → cold storage")

    if not result["cache_hit"]:
        step(f"Cache MISS — {sim_str} {'< ' + str(SIM_THRESH) if best_score else '(empty cache)'}")
        rag.metrics["misses"]           += 1
        rag.metrics["cold_storage_calls"] = rag.metrics.get("cold_storage_calls", 0) + 1

        # ── 4. Cold storage (BM25 + FAISS + RRF) ─────────────────────────────
        step("Routing to cold storage — BM25 sparse + FAISS dense + RRF fusion")
        t0 = time.perf_counter()
        search_result = rag.cold_storage.search(query)
        cold_ms = (time.perf_counter() - t0) * 1_000
        result["times"]["cold_ms"] = round(cold_ms, 2)

        raw_text = ""
        if search_result:
            _, raw_text = search_result
            result["retrieved_docs"]  = 3   # hybrid always returns top-3
            result["context_preview"] = raw_text[:800]
            step(f"Cold storage: top-3 docs retrieved via RRF ({round(cold_ms,1)} ms, {len(raw_text)} chars)")
        else:
            step("Cold storage: no results found in corpus")

        # ── 5. LLM answer generation ──────────────────────────────────────────
        if llm and raw_text:
            step(f"LLM call: {llm.provider}/{llm.model} — {len(raw_text)} char context")
            t0 = time.perf_counter()
            answer = llm.generate_answer(query, raw_text)
            llm_ms = (time.perf_counter() - t0) * 1_000
            result["times"]["llm_ms"] = round(llm_ms, 2)
            result["answer"] = answer
            step(f"LLM response: {round(llm_ms,1)} ms, {len(answer.split())} words")
        else:
            result["answer"] = raw_text[:1200] if raw_text else "No relevant information found."
            step("LLM unavailable — returning retrieved document text")

        # ── 6. Admit to cache ─────────────────────────────────────────────────
        rag.admit_to_cache(
            user_query   = query,
            query_vector = query_vector,
            raw_text     = raw_text,
            llm_answer   = result["answer"],
        )
        step(f"ARC admission — T1←({query_key[:8]}…), p={rag.cache.p}")
        rag.metrics["total_queries"] += 1

    # ── Finalize ───────────────────────────────────────────────────────────────
    result["times"]["total_ms"] = round((time.perf_counter()-t_wall)*1000, 2)
    step(f"Complete — {result['times']['total_ms']} ms total")

    result["arc_state"] = _arc_snapshot()
    result["metrics"]   = _metrics_snapshot()

    _query_log.append({
        "ts": time.strftime("%H:%M:%S"),
        "q":  query[:55],
        "hit": result["cache_hit"],
        "ms": result["times"]["total_ms"],
    })
    if len(_query_log) > 100:
        _query_log.pop(0)

    return result


def _which_tier(key: str) -> str:
    if key in rag.cache.t1: return "T1"
    if key in rag.cache.t2: return "T2"
    return "cache"


def _arc_snapshot() -> dict:
    c = rag.cache
    def items(d):
        return [{"key": k, "lbl": k[:10]} for k in d.keys()]
    return {
        "t1": items(c.t1), "t2": items(c.t2),
        "b1": items(c.b1), "b2": items(c.b2),
        "p": c.p, "capacity": c.c,
        "hot": len(c.t1) + len(c.t2),
        "ghost": len(c.b1) + len(c.b2),
    }


def _metrics_snapshot() -> dict:
    m = rag.metrics
    total = m["total_queries"]
    hits = m["hits"]
    misses = m["misses"]
    ghost_count = len(rag.cache.b1) + len(rag.cache.b2)
    ghost_bytes = ghost_count * 8  # SimHash: 8 bytes each
    ghost_full_bytes = ghost_count * 384 * 4  # full float32 vector

    # Cost estimation (Groq Llama-3.3-70B commercial rates)
    avg_ctx_tokens = 400
    avg_out_tokens = 150
    cost_per_call = (avg_ctx_tokens * 0.06 / 1e6) + (avg_out_tokens * 0.20 / 1e6)
    baseline_cost = total * cost_per_call
    actual_cost = misses * cost_per_call
    cost_saved = baseline_cost - actual_cost

    return {
        "total":   total,
        "hits":    hits,
        "misses":  misses,
        "hit_rate": round(hits / total * 100, 1) if total > 0 else 0.0,
        "cold_calls": m.get("cold_storage_calls", 0),
        "margin_rejects": m.get("margin_rejections", 0),
        "ttl_expires": m.get("ttl_expirations", 0),
        "b1": len(rag.cache.b1),
        "b2": len(rag.cache.b2),
        "p":  rag.cache.p,
        "hot": len(rag.cache.t1) + len(rag.cache.t2),
        "ghost_bytes": ghost_bytes,
        "ghost_full_bytes": ghost_full_bytes,
        "ghost_compression": f"{ghost_full_bytes // max(ghost_bytes,1)}x" if ghost_bytes > 0 else "—",
        "llm_avoided": hits,
        "llm_avoided_pct": round(hits / total * 100, 1) if total > 0 else 0.0,
        "est_cost_saved": round(cost_saved * 1000, 4),  # in millicents for readability
        "corpus_size": _get_corpus_size(),
    }


def _get_corpus_size() -> int:
    try:
        return rag.cold_storage.get_corpus_size()
    except Exception:
        return 0


def _reset_cache():
    rag.cache.t1.clear(); rag.cache.t2.clear()
    rag.cache.b1.clear(); rag.cache.b2.clear()
    rag.cache.p = 0
    rag.metrics.update({
        "hits": 0, "misses": 0, "total_queries": 0,
        "cold_storage_calls": 0, "margin_rejections": 0, "ttl_expirations": 0,
    })
    _query_log.clear()


# ─── HTTP Handler ─────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).parent

class _Handler(BaseHTTPRequestHandler):

    # ── routing ──────────────────────────────────────────────────────────────
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self._file(DEMO_DIR / "index.html", "text/html; charset=utf-8")
        elif p == "/test_queries.html":
            self._file(DEMO_DIR / "test_queries.html", "text/html; charset=utf-8")
        elif p == "/api/status":
            if not _ready.is_set():
                self._json({"ready": False, "initializing": True})
            elif rag is None:
                self._json({"ready": False, "error": _init_error})
            else:
                self._json({
                    "ready":    True,
                    "has_llm":  llm is not None,
                    "llm":      f"{llm.provider}/{llm.model}" if llm else None,
                    "corpus":   CORPUS_PATH,
                    "capacity": CACHE_CAP,
                    "threshold": SIM_THRESH,
                    "metrics":  _metrics_snapshot() if rag else {},
                    "arc_state": _arc_snapshot() if rag else {},
                })
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        p = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length)) if length else {}

        if p == "/api/query":
            if not _ready.is_set():
                self._json({"error": "System still initializing. Please wait a few seconds."}, 503)
                return
            if rag is None:
                self._json({"error": _init_error or "System failed to initialize."}, 500)
                return
            q = body.get("query", "").strip()
            if not q:
                self._json({"error": "Empty query."}, 400)
                return
            with _lock:
                result = _process_query(q)
            self._json(result)

        elif p == "/api/reset":
            if rag:
                with _lock:
                    _reset_cache()
            self._json({"ok": True})
        else:
            self.send_response(404); self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _file(self, path: Path, ctype: str):
        try:
            content = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", len(content))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404); self.end_headers()

    def _json(self, data: dict, status: int = 200):
        content = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(content)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        # Suppress routine access logs; only show errors
        if args and "200" not in str(args[1] if len(args) > 1 else args):
            super().log_message(fmt, *args)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))

    # Start initialization in background — server is immediately available
    init_thread = threading.Thread(target=_initialize, name="init", daemon=True)
    init_thread.start()

    server = HTTPServer(("0.0.0.0", PORT), _Handler)

    print()
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║      Vector-ARC RAG Demo Server           ║")
    print(f"  ║      http://localhost:{PORT}               ║")
    print("  ║                                           ║")
    print("  ║  System loading in the background.        ║")
    print("  ║  The browser will show 'initializing'     ║")
    print("  ║  until the embedding model is ready.      ║")
    print("  ╚═══════════════════════════════════════════╝")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
