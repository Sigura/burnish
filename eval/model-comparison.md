# Model comparison (November 2026)

Eval set: `eval/sentences.tsv`, 104 English sentences with one reference correction each.
Hardware: RTX 4090, llama-cpp-python 0.3.36, `n_gpu_layers=-1`, `flash_attn=true`, KV cache Q8_0 (`type_k/type_v = 8`).

## Models

| model | arch | params total / active | quant | ~bpw | file size | gen tok/s | prompt tok/s | cold load |
|---|---|---|---|---|---|---|---|---|
| **gemma4:26b-a4b** | MoE, 128 experts (8 active) | 25.2B / **~4B** | QAT Q4_0 | 4.6 | 13.45 GiB | **148** (115¹) | 4470 (7200¹) | 19.7 s |
| **gemma4:12b** | dense | ~12B | QAT Q4_0 | 4.6 | 6.50 GiB | 86 | **6112** | 10.0 s |
| gemma4:31b | dense | ~31B | QAT Q4_0 | 4.6 | 16.4 GiB | 42 | 2964 | 24.1 s |
| qwen3.8:27b | hybrid: 16 attention + 48 linear-attention layers | 27B | GSQ-RCO IQ3_S (mixed) | **3.5** | 10.96 GiB | 58 (54¹) | 1736 (2150¹) | 15.9 s |
| qwen3:14b | dense | 14B | Q4_K_M | 4.9 | 8.38 GiB | 88 | 5719 | 12.0 s |
| gemma3:12b | dense | 12B | Q4_K_M | 4.8 | 6.80 GiB | 86 | 5155 | 10.2 s |
| qwen3:8b | dense | 8B | Q4_K_M | 4.9 | 4.68 GiB | — | — | — |
| gemma3:4b | dense | 4B | Q4_K_M | ~5 | 2.32 GiB | — | — | — |

Speed: llama.cpp microbenchmark, 1500-token prompt and 256 generated tokens, warm model. Cold load is the
first load from disk. qwen3:8b and gemma3:4b were not benchmarked.

¹ Re-bench on 2026-11-13 on nvidia-smi GPU 2 (RTX 4090) while another process held 7.7 GB on that card. The
other speed numbers come from an earlier run on a card whose identity is unclear: `CUDA_DEVICE_ORDER` was
unset, so CUDA device indices did not match nvidia-smi's.

## Quality and latency in burnish

| model | hybrid exact | llm-only exact | all-mode exact | all-mode over-correction | hybrid p50 | all-mode p95 |
|---|---|---|---|---|---|---|
| **gemma4:26b-a4b** | **99.0%** | **99.0%** | 79.8% | 17.3% | 0.12 s | 0.34 s |
| **gemma4:12b** | **99.0%** | **99.0%** | 66.3% | 29.8% | 0.16 s | 0.53 s |
| gemma4:31b | 98.1% | 98.1% | 77.9% | 19.2% | 0.28 s | 0.93 s |
| qwen3.8:27b² | 97.1% | 97.1% | 70.2% | 27.9% | 0.29 s | 0.76 s |
| gemma3:12b | 95.2% | 95.2% | 73.1% | 24.0% | 0.15 s | 0.49 s |
| qwen3:14b | 92.3% | 92.3% | **91.3%** | **7.7%** | 0.21 s | 0.52 s |
| gemma3:4b | 82.7% | 81.7% | 73.1% | 22.1% | 0.11 s | 0.31 s |
| qwen3:8b | 81.7% | 80.8% | 88.5% | 6.7% | 0.15 s | 0.43 s |
| *LanguageTool only* | *26.9%* | — | — | *0%* | *0.08 s* | — |

Source: `eval/results/20261113-091452.json` (`make eval`). Modes:
- **hybrid** and **llm-only** run fix mode with and without LanguageTool.
- **all** adds the polish step, so it goes beyond the minimal reference corrections. Its exact-match score is
  therefore lower, and over-correction counts outputs that touched more words than the reference did.

² Measured on 2026-11-13 on nvidia-smi GPU 2 (RTX 4090), after burnish began rendering the chat template
with `enable_thinking=False` (`src/burnish/engine.py`). Before that fix the model ignored `/no_think` and wrote
its reasoning into every reply. All 104 rewrites were rejected, and the score was only the LanguageTool
fallback: 26.9% exact. Called directly through its template with thinking off, it scores 103/104. Its 3
burnish misses:
- `lgtm, let's` became `lgtm; let's`.
- `have finish` became `have finished` where the reference is `finished`.
- `don't need no help` was left unchanged.

## Thinking on vs off (direct template calls)

| model | thinking | exact | s/sentence | tokens/sentence |
|---|---|---|---|---|
| gemma4:26b-a4b | off | 102/104 | 0.12 | 9 |
| gemma4:26b-a4b | on | 96/104 | 2.41 | 300 |
| qwen3.8:27b | off | 103/104 | 0.27 | 9 |
| qwen3.8:27b | on | 99/104 | 2.21 | 116 |

## Takeaways

- **Speed depends on active parameters, not file size.** gemma4:26b-a4b computes about 4B parameters per token,
  so it generates faster than the 6.5 GiB dense gemma4:12b.
- **Qwen3.8 has the smallest bits per weight but is the slowest per GiB.** IQ3_S decodes weights slowly. Its
  hybrid layers also stop llama-cpp-python from reusing the prompt prefix, so the whole prompt is recomputed
  for every sentence.
- **Qwen3.8 doesn't beat the Gemma 4 models.** With thinking off it reaches 97% in fix mode at about twice
  the latency of gemma4:26b-a4b, and in all mode it over-corrects more (27.9% vs 17.3%).
- **Fix mode is saturated at about 99%.** The models only separate in all mode, where the Qwen3 models
  over-correct the least.
- **Thinking hurts on this task.** It was 20× slower and less accurate.
- **KV cache is not the difference.** At `n_ctx=8192` with a Q8_0 cache, gemma4:26b-a4b needs about 0.3–1 GB
  depending on how the sliding-window layers are allocated, and qwen3.8:27b about 0.3 GB. Qwen3.8 also keeps
  about 150 MB of fixed f32 recurrent state, which `type_k`/`type_v` do not quantize.
