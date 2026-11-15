#!/usr/bin/env bash
# Re-quantizes gemma4:26b-a4b to Q4_0 with an imatrix from burnish-style fix prompts and measures the KL divergence
# against bf16 on held-out fix prompts for Google's QAT file, the wikitext-imatrix Q4_0 and the new file.
# Runs on the CPU only, so it needs no GPU. Every step skips work it already finished.
# Usage: quant/run_fix.sh   Env: THREADS (default 48), WORK (scratch dir), MODELS (burnish models dir).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
export WORK="${WORK:-$HOME/.local/share/burnish/quant-work}"
MODELS="${MODELS:-$(dirname "$WORK")/models}"
THREADS="${THREADS:-48}"
PY="$WORK/venv/bin/python"
BURNISH_PY="$HERE/../.venv-devbox/bin/python"
# GPUS (nvidia-smi indices, e.g. 2,3) speeds up the imatrix; the default runs everything on the CPU.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPUS:-}"

echo "== checking prerequisites"
[[ -x "$WORK/llama.cpp/build/bin/llama-imatrix" ]] || { echo "run quant/build_llamacpp.sh first" >&2; exit 1; }
[[ -f "$WORK/gguf/gemma4-26b-a4b-qat-bf16.gguf" ]] || { echo "run quant/run_all.sh first (bf16 GGUF)" >&2; exit 1; }
[[ -x "$BURNISH_PY" ]] || { echo "missing burnish venv at $BURNISH_PY" >&2; exit 1; }
GOOGLE="$MODELS/gemma4-26b-a4b/gemma-4-26B_q4_0-it.gguf"
WIKI="$MODELS/gemma4-26b-a4b-imat/gu-q4_0_dn-q4_0.gguf"
FIX="$WORK/out/gemma-4-26B-A4B-it-qat-q4_0-imat-fix.gguf"
for f in "$GOOGLE" "$WIKI"; do [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }; done

step() { echo; echo "== $*"; "$PY" "$HERE/rco_lite.py" "$@"; }
burnish_step() { echo; echo "== $*"; "$BURNISH_PY" "$HERE/rco_lite.py" "$@"; }

[[ -f "$WORK/data/fix.test.jsonl" ]] || burnish_step corpus
[[ -f "$WORK/imatrix-fix.gguf" ]] || step imatrix --corpus fix --threads "$THREADS"
[[ -f "$FIX" ]] || step quantize-fix
[[ -f "$WORK/kld/bf16-fix.npy" ]] || burnish_step kld-fix --threads "$THREADS"
for m in "$GOOGLE" "$WIKI" "$FIX"; do
    burnish_step kld-fix "$m" --threads "$THREADS"
done

echo
echo "== done; results in $WORK/out/kld-fix-*.json"
