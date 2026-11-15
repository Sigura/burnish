# Model recommendations for this host (November 2026)

Host: 3× RTX 4090 (nvidia-smi 0, 2, 3) + 1× RTX 3090 Ti (nvidia-smi 1), 24 GB each; 128 CPU threads; 503 GB RAM.
GPU work runs on GPUs 2 and 3 (48 GB): GPU-heavy runs have crashed the host three times, cause unknown.
Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`; otherwise CUDA numbers the GPUs differently from nvidia-smi.

## Winner for burnish (short-text correction): gemma4:26b-a4b, QAT Q4_0

Config:

| setting | value |
|---|---|
| model | `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, 13.45 GiB |
| architecture | MoE, 128 experts, 8 active (~4B active params); 30 layers: 25 sliding-window (1024) + 5 global |
| llama.cpp options | `n_gpu_layers=-1`, `flash_attn=true`, `type_k=type_v=8` (Q8_0 KV), `n_ctx=8192` |
| VRAM | ~15–16 GB: model + 935 MiB KV + 533 MiB compute |
| chat template | thinking off (`enable_thinking=False`) |

Why it won (details in `model-comparison.md`):
- **Speed.** 115–148 tok/s generation and 4.5–7k tok/s prompt processing. Generation speed depends on active
  params, not file size: the dense 31B does 42 tok/s and the dense 12B 86 tok/s.
- **Quality.** 99% exact on the fix eval, which is the eval's ceiling. All mode over-corrects 17%, the lowest of
  the Gemma models.
- **Quant.** QAT Q4_0 suits it best: the model was trained for that grid, and Q4_0 has the fastest kernels.
  Our smaller RCO-lite builds (10–11 GiB) score the same in burnish, but don't fit a 12 GB card at
  `n_ctx=8192` yet.
- **12 GB fallback.** gemma4:12b QAT Q4_0: 6.5 GiB, 99%, 86 tok/s.

## Lessons that carry over to other workloads

- **Prefer MoE models with few active params.** Dense models of 27B and up are 2–3× slower here.
- **i-quants save VRAM but cost speed.** IQ3_S and similar decode more slowly; Q4_0 and Q4_K are faster per GB.
  Imatrix-weighted Q4_0 is much closer to bf16 than plain Q4_0 (KLD 0.21 vs 0.35).
- **Hybrid SSM models (Qwen3.5+ architecture): avoid llama-cpp-python.** It can't reuse the prompt prefix for
  them, so it recomputes the whole prompt on every call. Use `llama-server` and verify its context
  checkpointing.
- **Sliding-window models:** llama-cpp-python sizes their KV cache for the full context. `swa_full=False`
  should save most of it (not tested yet).
- **Thinking:** it made short copy-edits 20× slower and less accurate. Test it separately for coding.
- **Card size:** each card has 24 GB, so anything over about 20 GB needs a multi-GPU split or MoE experts in
  RAM (`--n-cpu-moe`, `--fit`).

## Coding model suggestions

Scores are vendor-reported. None of these models has been measured on this host yet.

| | model | arch | coding score | fit on GPUs 2+3 (48 GB) | why |
|---|---|---|---|---|---|
| **1. main** | Qwen3.8-27B | dense hybrid (16 attention + 48 linear-attention layers) | SWE-bench Pro 61.7% | Q6_K/Q8_0, 22–29 GB, on one or two GPUs | nearly matches the 125B Flash-Next and fits comfortably; dense, so behavior is predictable |
| **2. fast agent** | Qwen3-Coder-Next 80B-A3B | MoE, 3B active, linear attention | SWE-bench Verified ~71% (OpenHands) | IQ3/Q3_K (~35 GB), or Q4_K_M (~46 GB) with `--n-cpu-moe 4–8` | very fast decode for long agent loops |
| **3. stretch** | Qwen3.8-Flash-Next 125B-A6B | MoE, 6B active | SWE-bench Pro 62.5% (best open) | IQ4_XS is ~94 GB: experts in RAM, attention + KV on GPUs 2+3 | highest score, but slower with experts on the CPU; needs a newer llama.cpp than `b12020`; Qwen Community License |

Skip:
- dense 120B+ (Devstral 2 is ~62 GB at Q4)
- DeepSeek V4.1-Flash (552B+)
- Gemma 4 26B-A4B for coding (excellent for burnish, weak at agentic coding)

Notes:
- We already have Qwen3.8-27B as GSQ-RCO IQ3_S (11 GB). It's fine for a first try, but use Q5 or higher for
  coding.
- Burnish's thinking fix (`enable_thinking=False`) is right for copy-edits. Coding agents probably want it on.

Next step: build `llama-server` (llama.cpp `b12020` or newer) and download candidates 1 and 2. Then measure
generation and prompt-processing speed at 32k context on GPUs 2+3, plus a small coding eval (an Aider
polyglot subset or tasks from our own repos).

## Sources

- [SWE-bench Pro leaderboard (Morph, Nov 2026)](https://www.morphllm.com/swe-bench-pro)
- [Best open-weight coding models to self-host (Digital Applied)](https://www.digitalapplied.com/blog/best-open-weight-coding-models-self-host-hardware-match-2026)
- [Best AI model for coding, Nov 2026 (Morph)](https://www.morphllm.com/best-ai-model-for-coding)
- [Unsloth: Qwen3.8-Flash-Next](https://unsloth.ai/docs/models/qwen3.8-next)
- [Faros: best open-weight models for coding](https://www.faros.ai/blog/open-weight-models)
- [MindStudio: DeepSeek V4.1 Flash local](https://www.mindstudio.ai/blog/deepseek-v4-1-flash-local/)
