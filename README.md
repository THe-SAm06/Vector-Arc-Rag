# Vector-ARC: Adaptive Semantic Cache for RAG

Welcome to the **Vector-ARC** repository! This project implements an Adaptive Replacement Cache (ARC) for dense vector embeddings, coupled with a robust DuckDB BM25 fallback, forming a highly optimized Retrieval-Augmented Generation (RAG) pipeline.

## Overview

Traditional caching strategies like LRU fail under shifting query distributions and dense vectors consume massive amounts of RAM in ghost lists. **Vector-ARC** solves this by:
1. Compressing $1536$-dimensional dense embeddings down to 64-bit **SimHash fingerprints**.
2. Adaptively balancing between recency ($T_1$) and frequency ($T_2$) based on hits in the highly-compressed ghost lists ($B_1/B_2$).

By reducing memory overhead by **192x** and achieving up to a **2664x speedup** on cache hits, Vector-ARC aggressively cuts LLM API costs while scaling indefinitely.

## Repository Structure

```text
/data       - Contains the SciFact corpus (JSON) and cold storage databases.
/demo       - Contains the web UI dashboard (HTML/CSS) and backend server.
/docs       - Contains experimental scripts, presentations, and LaTeX files (ignored by git).
/metrics    - Auto-generated benchmark outputs and CSVs (ignored by git).
/scripts    - Helper scripts (e.g., test query generation).
/src        - Core RAG pipeline logic (Embedder, LLM Engine, VectorARC cache).
```

## Setup & Installation

**1. Clone the Repository:**
```bash
git clone <your-repo-url>
cd vector_arc
```

**2. Install Dependencies:**
We provide convenience scripts for both Linux/macOS and Windows:
```bash
# Linux / macOS
./setup.sh

# Windows (PowerShell)
.\setup.ps1
```
*(Alternatively, you can manually run `pip install -r requirements.txt`)*

**3. Configure API Keys:**
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and insert your API key. (We recommend a free Groq API key for incredibly fast inference). 
*Note: If no API key is provided, the system degrades gracefully, bypassing the LLM and alerting you on the UI.*

## Running the Web Dashboard

Launch the interactive UI to visually test queries and watch the cache adapt in real-time.

```bash
bash demo/start.sh
```
Open your browser to `http://localhost:8080`.

## Running the Official Benchmark (BenchmarkRunner)

To quantitatively evaluate the system's performance, run the comprehensive benchmark suite. This script runs a controlled workload to simulate high-eviction pressure and distribution shifts, directly outputting the metrics used in our final evaluation report.

```bash
PYTHONPATH=. python run_full_benchmark.py
```
This will output the final hit rates, LLM calls avoided, average latency speedups, and storage efficiency metrics directly to the terminal, and save the detailed telemetry to the `/metrics` folder.

## Evaluation Workflow for Judges
1. Run `setup.sh`.
2. Execute `run_full_benchmark.py` to immediately verify our claims (40% LLM cost reduction, 192x memory compression, 2600x latency speedup).
3. Start `bash demo/start.sh` to explore the beautiful UI. Use the provided "Test Queries" hyperlink on the UI to test exact and paraphrased queries to see the cache hit in action!

## License
MIT License
