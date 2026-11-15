#!/usr/bin/env bash
# Benchmarks a GGUF with llama-bench (empty context and 32k depth) and llama-server (one ~28k-token
# prompt, then the same prompt plus a suffix to check prefix reuse). GPUs 2 and 3 only.
# Usage: eval/bench_llama_server.sh MODEL.gguf [extra llama-server/llama-bench args...]
set -euo pipefail

MODEL="${1:?usage: $0 MODEL.gguf [extra args]}"; shift
EXTRA=("$@")
BIN="${LLAMA_BIN:-$HOME/.local/share/burnish/quant-work/llama.cpp/build/bin}"
CTX="${CTX:-32768}"
PORT="${PORT:-8089}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPUS:-2,3}"

echo "== checking prerequisites"
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }
for b in llama-bench llama-server; do
    [[ -x "$BIN/$b" ]] || { echo "missing $BIN/$b (build with quant/build_llamacpp.sh + target $b)" >&2; exit 1; }
done
command -v python3 >/dev/null || { echo "missing python3" >&2; exit 1; }

echo "== llama-bench: pp2048 / tg128 at depth 0 and $CTX"
"$BIN/llama-bench" -m "$MODEL" -fa on -ctk q8_0 -ctv q8_0 -p 2048 -n 128 -d "0,$CTX" -r 2 "${EXTRA[@]}"

echo "== llama-server at -c $CTX"
LOG="$(mktemp)"
"$BIN/llama-server" -m "$MODEL" -c "$CTX" -fa on -ctk q8_0 -ctv q8_0 -ngl 99 --jinja \
    --port "$PORT" --host 127.0.0.1 "${EXTRA[@]}" >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; rm -f "$LOG"' EXIT
for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { tail -40 "$LOG"; exit 1; }
    sleep 1
done
grep -E "CUDA[0-9] (model|KV|RS|compute) buffer" "$LOG" || true

python3 - "$PORT" "$(dirname "$0")/.." <<'EOF'
import json, pathlib, sys, urllib.request
port, root = sys.argv[1], pathlib.Path(sys.argv[2])

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

text = "\n\n".join(f"# {p}\n{p.read_text()}" for p in sorted(root.glob("src/**/*.py")) + sorted(root.glob("quant/*.py")))
toks = post("/tokenize", {"content": text})["tokens"]
while len(toks) < 28000:
    toks += toks
toks = toks[:28000]
tail = post("/tokenize", {"content": "\n\nSummarize what the code above does in five bullet points."})["tokens"]
for name, prompt in [("cold 28k", toks + tail), ("same prefix + suffix", toks + tail + tail)]:
    t = post("/completion", {"prompt": prompt, "n_predict": 256, "temperature": 0, "cache_prompt": True})["timings"]
    print(f"{name:22} prompt {t['prompt_n']:6d} tok {t['prompt_per_second']:8.1f} tok/s | "
          f"gen {t['predicted_n']:4d} tok {t['predicted_per_second']:6.1f} tok/s")
EOF
echo "== VRAM (nvidia-smi, GPUs $CUDA_VISIBLE_DEVICES)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES"
