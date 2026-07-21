#!/bin/bash
# ─── Vector-ARC Demo Server Launcher ──────────────────────────────────────────
# Uses the conda env that has all ML dependencies:
#   sentence-transformers, numpy, rank_bm25, openai, dotenv, torch
#
# Usage:
#   bash demo/start.sh            # starts on port 8080
#   PORT=9090 bash demo/start.sh  # custom port
# ──────────────────────────────────────────────────────────────────────────────

PYTHON="/home/sam/miniconda3/envs/tf_gpu_conda/bin/python3"

# Fallback to system python3 if the conda env is missing
if [ ! -f "$PYTHON" ]; then
  PYTHON="python3"
fi

# Must run from the project root (so src/ and data/ are importable)
cd "$(dirname "$0")/.."

echo ""
echo "  ▶ Vector-ARC Demo"
echo "    Python: $PYTHON"
echo "    Open:   http://localhost:${PORT:-8080}"
echo ""

"$PYTHON" demo/server.py
