#!/usr/bin/env bash
# Downloads the bf16 QAT checkpoint that RCO-lite quantizes from. Resumable: re-run after an interruption.
set -euo pipefail

WORK="${WORK:-$HOME/.local/share/burnish/quant-work}"
REPO="${SOURCE_REPO:-google/gemma-4-26B-A4B-it-qat-q4_0-unquantized}"
DEST="$WORK/src/${REPO##*/}"
FILES=(config.json generation_config.json chat_template.jinja tokenizer.json tokenizer_config.json
       processor_config.json model.safetensors.index.json
       model-00001-of-00002.safetensors model-00002-of-00002.safetensors)

echo "== checking prerequisites"
command -v curl >/dev/null || { echo "missing: curl" >&2; exit 1; }
mkdir -p "$DEST"

auth=()
[[ -n "${HF_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

for f in "${FILES[@]}"; do
    echo "== $f"
    curl -fL --retry 5 --retry-delay 10 -C - "${auth[@]}" -o "$DEST/$f" \
        "https://huggingface.co/$REPO/resolve/main/$f"
done

echo "== done: $DEST"
du -sh "$DEST"
