#!/usr/bin/env bash
# Runs the RCO-lite pipeline end to end. Every step skips work it already finished, so it can be re-run.
# Usage: quant/run_all.sh [GPU ...]   (nvidia-smi indices for the sweep; default: 2)
# Env: BUDGETS="10 10.5 11" (GiB targets), WORK (scratch dir).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HOME/.local/share/burnish/quant-work}"
BUDGETS="${BUDGETS:-10 10.5 11}"
GPUS=("${@:-2}")
PY="$WORK/venv/bin/python"
BURNISH_PY="$HERE/../.venv-devbox/bin/python"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "== checking prerequisites"
[[ -x "$WORK/llama.cpp/build/bin/llama-quantize" ]] || { echo "run quant/build_llamacpp.sh first" >&2; exit 1; }
[[ -f "$WORK/src/gemma-4-26B-A4B-it-qat-q4_0-unquantized/model-00002-of-00002.safetensors" ]] \
    || { echo "run quant/fetch_source.sh first" >&2; exit 1; }
[[ -x "$BURNISH_PY" ]] || { echo "missing burnish venv at $BURNISH_PY" >&2; exit 1; }

step() { echo; echo "== $*"; "$PY" "$HERE/rco_lite.py" "$@"; }

[[ -f "$WORK/data/wiki.test.raw" ]] || step data
[[ -f "$WORK/gguf/gemma4-26b-a4b-qat-bf16.gguf" ]] || step convert
[[ -f "$WORK/imatrix.gguf" ]] || step imatrix
[[ -f "$WORK/kld/bf16.kld" ]] || step kld-base
step quantize
step sweep --gpus "${GPUS[@]}"

for b in $BUDGETS; do
    step allocate --budget-gib "$b"
    step build --budget-gib "$b" --gpus "${GPUS[0]}"
    echo "== VRAM fit for ${b} GiB plan"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$BURNISH_PY" "$HERE/fit_check.py" \
        "$WORK/out/gemma-4-26B-A4B-it-qat-rcolite-${b}g.gguf"
done

echo
echo "== done; outputs in $WORK/out"
