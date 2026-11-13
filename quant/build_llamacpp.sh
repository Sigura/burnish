#!/usr/bin/env bash
# Builds the llama.cpp tools used by the RCO-lite pipeline (quantize, imatrix, perplexity)
# and a Python venv for convert_hf_to_gguf.py. Everything lives under $WORK.
set -euo pipefail

WORK="${WORK:-$HOME/.local/share/burnish/quant-work}"
TAG="${LLAMA_CPP_TAG:-b12020}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
SRC="$WORK/llama.cpp"
VENV="$WORK/venv"

echo "== checking prerequisites"
for tool in git uv gcc g++; do
    command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "nvcc not found under $CUDA_HOME" >&2; exit 1; }
mkdir -p "$WORK"

echo "== llama.cpp source at $TAG"
if [[ ! -d "$SRC/.git" ]]; then
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
git -C "$SRC" fetch -q --tags
git -C "$SRC" -c advice.detachedHead=false checkout -q "$TAG"

echo "== python venv (cmake, ninja, converter requirements)"
[[ -x "$VENV/bin/python" ]] || uv venv -q "$VENV" --python 3.12
VIRTUAL_ENV="$VENV" uv pip install -q cmake ninja -r "$SRC/requirements/requirements-convert_hf_to_gguf.txt"

echo "== configuring CUDA build (sm_86, sm_89)"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
cmake -S "$SRC" -B "$SRC/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF

echo "== building tools"
cmake --build "$SRC/build" -j "$(nproc)" --target llama-quantize llama-imatrix llama-perplexity

echo "== done"
ls -la "$SRC/build/bin/llama-quantize" "$SRC/build/bin/llama-imatrix" "$SRC/build/bin/llama-perplexity"
