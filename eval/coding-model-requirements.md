# Coding model requirements for this host (November 2026)

What a local coding model must meet to be worth serving here, derived from the measurements in
`model-comparison.md`, `coding-bench.md` and llm_coder's `~/llm_coder/llm/results/two-model-2026-11-13/report.md`.
"Measured" marks a number from this host; everything else is a target or a vendor figure.

## Workload

- Agentic coding (OpenCode-style): many short turns over one long, growing context, plus tool calls.
- The prompt prefix is shared between turns, so **prefix reuse** matters as much as raw prefill speed.
- Decode speed sets the turn latency; long agent loops need a fast model, not only a strong one.

## Hardware budget

| resource | value | rule |
|---|---|---|
| GPUs | 3× RTX 4090 (nvidia-smi 0, 2, 3) + 1× RTX 3090 Ti (1), 24 GB each | always `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| default GPUs for heavy work | 2, 3 (48 GB) | GPU-heavy runs have crashed the host 3×, cause unknown. GPUs 0, 1 only with user approval |
| shared GPUs | 0, 1: DeepSeek on 8080 (llm_coder), 2: Qwen3.6 on 8082 | both unload after 600 s idle and reload on request: coordinate before using these cards |
| CPU | 2× EPYC 7601 (Zen1), 128 threads | slow for MoE experts in RAM, very slow for i-quants there |
| RAM | 503 GB | room for experts offload, at a speed cost |
| network | ~6–10 MB/s, shared between sessions | a 30 GB download is ~1 h; coordinate large downloads |

## Requirements

### Architecture

| requirement | target | why (evidence) |
|---|---|---|
| type | **MoE with ≤ ~6B active params** | decode speed follows active params: 3B-active MoE 113–156 tok/s vs dense 27B 38 tok/s (measured) |
| dense models | only ≤ ~30B, as a quality fallback | dense 27B Q6_K: 38 tok/s on one 4090 (measured); dense 120B+ doesn't fit at a useful quant |
| attention | hybrid linear/full attention is fine **in llama-server** | prefix reuse worked on both Qwen hybrids: 18 of 28k tokens recomputed (measured). llama-cpp-python can't reuse the prefix for these models |
| sliding-window models | set `swa_full=False` (llama-cpp-python) or default `--swa-full` off (llama-server) | gemma4: SWA KV cache 850 → 159 MiB at 8k (measured) |
| context | ≥ 32k usable, 64k preferred | agent sessions; Qwen3.6 passes a 54k needle test at 64k (llm_coder) |
| tool calling | working chat template with tool calls (`--jinja`) | OpenCode depends on it; Qwen3.6 passes the healthcheck (llm_coder) |
| thinking | switchable per request | thinking made copy-edits 20× slower; for coding, test both |
| license | open weights usable for local work | Qwen3.8-Flash-Next is under the Qwen Community License; check before adopting |

### Quantization

| requirement | target | why (evidence) |
|---|---|---|
| bits | ≥ ~4.5 bpw for a main coding model (Q4_K/Q5_K/Q6_K); 3–3.5 bpw only for a "fast agent" tier after a quality check | coding is more sensitive than copy-edits; burnish's eval can't tell quants apart, so it isn't evidence |
| placement | **fully in VRAM** for i-quants (IQ2/IQ3/IQ4_XS) | i-quants ~5× slower than K-quants with experts in CPU RAM on Zen1 (llm_coder) |
| experts in RAM (`--n-cpu-moe`) | only with K-quants or Q4_0 | same reason |
| kernel speed | prefer Q4_0/Q4_K/Q6_K for speed per GB | IQ3_S decodes slower per GiB than Q4_0/Q4_K (burnish) |
| KV cache | Q8_0 (`-ctk q8_0 -ctv q8_0`) | halves KV memory; no quality issue seen in burnish |
| custom mixes | per-tensor mixes via `llama-quantize --tensor-type` regex (works in b12020) or split-shard RCO | imatrix-weighted Q4_0 beat plain Q4_0 on KLD 0.21 vs 0.35 for gemma4 |
| validation | KLD vs bf16 on in-domain data (chat-formatted code episodes), not wikitext | wikitext inflates KLD for instruct models (bf16 PPL 551 for gemma4) |

### VRAM budget

| target | budget | fits today (measured) |
|---|---|---|
| one card | ≤ ~22 GB incl. KV and CUDA context | Qwen3.8-27B Q6_K at 32k: ~22.0 GB; Qwen3.6-35B-A3B Q4_K_XL at 64k: 22.3 GB |
| two cards (2,3) | ≤ ~44 GB | Coder-Next IQ3_S at 32k: ~30.2 GB; Q4_K_XL (46.2 GiB file) would need `--n-cpu-moe` |
| 12 GB fallback card (burnish only) | ≤ ~11.5 GB | gemma4:26b-a4b RCO-lite 10 GiB with `swa_full=False`: ~11.3–11.5 GB (estimated with CUDA context) |

### Performance targets (proposed)

| metric | target | measured so far |
|---|---|---|
| decode, empty context | ≥ 80 tok/s | Qwen3.6 134–156, Coder-Next IQ3_S 113, Qwen3.8-27B 38 |
| decode at 32k context | ≥ 50 tok/s | Coder-Next 72–76, Qwen3.8-27B 33 |
| prefill at 32k | ≥ 1500 tok/s | Qwen3.6 7073, Qwen3.8-27B 1541, Coder-Next 1537 |
| prefix reuse | ≤ 1% of the prompt recomputed on a repeat turn | both Qwen hybrids: 18 of 28k tokens |
| coexisting with DeepSeek | ≤ 10% decode loss when both generate | Qwen3.6: −10% (llm_coder) |

### Quality gate (open)

- No coding quality number on this host yet. Vendor figures: Qwen3.6-35B-A3B SWE-bench Verified 73.4,
  Qwen3-Coder-Next 70.6, Qwen3.8-27B SWE-bench Pro 61.7.
- Needed: a coding eval shared with llm_coder, so quants and models can be compared on the same set. Candidate
  set: llm_coder's 155 test episodes (`llm_coder/llm/quant-work/data`), plus an OpenCode e2e task (Qwen3.6 passes
  in repos under `~`, fails in very deep `/tmp/...` paths).

## Serving defaults

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 \
llama-server -m MODEL.gguf -c 32768 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja
```

Benchmark: `eval/bench_llama_server.sh MODEL.gguf` (`GPUS=` to override, default 2,3).

## Candidates against these requirements

| model | arch | meets | misses / open |
|---|---|---|---|
| Qwen3.6-35B-A3B UD-Q4_K_XL | MoE 3B active | speed, one card, tool calls, 64k | quality gate |
| Qwen3-Coder-Next 80B-A3B UD-IQ3_S | MoE 3B active, hybrid | speed, prefix reuse, fits 2 cards | bits below the main-model target; i-quant (GPU only); quality gate; 2× 4090 numbers not measured |
| Qwen3-Coder-Next Q4_K_XL / RCO ~3.5 bpw | same | — | in progress at llm_coder |
| Qwen3.8-27B UD-Q6_K | dense hybrid | quant quality, one card, prefix reuse | decode 33–38 tok/s, below target |
| Qwen3.8-Flash-Next 125B-A6B | MoE 6B active | vendor score (SWE-bench Pro 62.5) | ~94 GB at IQ4_XS: experts in RAM with an i-quant (slow here); needs newer llama.cpp; license |
| Gemma 4 26B-A4B | MoE 4B active | speed | weak at agentic coding: excluded |
