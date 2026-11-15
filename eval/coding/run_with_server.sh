#!/usr/bin/env bash
# Starts llama-server for MODEL.gguf, runs the coding eval against it, stops the server.
# Usage: eval/coding/run_with_server.sh MODEL.gguf LABEL [run_coding_eval.py args...]
# GPUs: inside `devbox job --gpus device=2,3` leave GPUS unset; on a host with all GPUs visible set GPUS=2,3.
set -euo pipefail

MODEL="${1:?usage: $0 MODEL.gguf LABEL [eval args]}"
LABEL="${2:?usage: $0 MODEL.gguf LABEL [eval args]}"
shift 2
BIN="${LLAMA_BIN:-$HOME/.local/share/burnish/quant-work/llama.cpp/build/bin}"
CTX="${CTX:-32768}"
PORT="${PORT:-8089}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ -n "${GPUS:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPUS"

echo "== checking prerequisites"
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }
[[ -x "$BIN/llama-server" ]] || { echo "missing $BIN/llama-server (quant/build_llamacpp.sh)" >&2; exit 1; }
command -v python3 >/dev/null || { echo "missing python3" >&2; exit 1; }
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { echo "port $PORT already in use" >&2; exit 1; }

echo "== llama-server on port $PORT, -c $CTX"
LOG="$(dirname "$0")/results/server-$LABEL.log"
mkdir -p "$(dirname "$LOG")"
"$BIN/llama-server" -m "$MODEL" -c "$CTX" -fa on -ctk q8_0 -ctv q8_0 -ngl 99 --jinja \
    --port "$PORT" --host 127.0.0.1 >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { tail -40 "$LOG"; exit 1; }
    sleep 1
done

echo "== coding eval ($LABEL)"
python3 "$(dirname "$0")/run_coding_eval.py" --url "http://127.0.0.1:$PORT" --label "$LABEL" "$@"
