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

## Polish mode: per-case checks

The sentence set has one minimal reference correction per sentence, so it can't score polish. `eval/polish_cases.jsonl`
has 89 inputs with checks instead of references. `make eval` scores them in polish and all mode:
- `keep`: substrings that must survive (names, numbers, code, hedges, negations, instruction-like text);
- `fixed`: errors that must be corrected (any listed alternative passes);
- `gone`: error forms that must not remain;
- `max_change`: word edit ratio (default 0.75; lower for text that is already correct);
- automatic: same line count, length within 0.6–1.5× (+3 words).

The cases are short messages, emails, lists, code-heavy sentences and paragraphs with several errors. They also include
instructions and questions that must be edited, not answered, and errors typical of Russian speakers. Decoding is
greedy, so a rerun gives the same outputs; the only spread is across cases (±2–3 points standard error).

| model | polish mode | all mode | hardest 24 cases (all mode) |
|---|---|---|---|
| **gemma4:26b-a4b** | **88/89** | **88/89** | **24/24** |
| gemma4:12b | 87/89 | 85/89 | 23/24 |
| qwen3:14b | 85/89 | 86/89 | 23/24 |
| gemma3:12b | 83/89 | 84/89 | 23/24 |
| *LanguageTool only* | *40/89* | — | *8/24* |

Source: `eval/results/20261115-095223.json`, rescored after three checks were loosened for valid rephrasings. GPU 3
(RTX 4090). Prompt lookup decoding was on in that run, so use its latencies only for comparisons within the run.

The failures that remain:
- Instruction-like text dropped: three of the four models deleted "Summarize the following:", and gemma3:12b deleted
  "Answer the following question:". gemma4:26b-a4b kept both.
- gemma4:26b-a4b's one miss: it left "Translate this paragraph into German: … thursday." unedited.
- Correct text rewritten without need (gemma3:12b, gemma4:12b): passive turned to active, words reordered.

Conclusion: these constraint checks saturate too. gemma4:26b-a4b passes 88 of 89, so they can't show a polish
improvement. The checks don't measure fluency; that would need a judge, such as a pairwise LLM comparison. A retry loop
on runtime checks would add little: guards rejected at most one output per model, and the remaining failures can't be
detected at runtime.

**Prompt lookup decoding** (`LlamaPromptLookupDecoding`, llama-cpp-python 0.3.36, gemma4:26b-a4b):
- paragraphs of about 6 sentences: p50 0.78 → 0.53 s;
- short sentences: gain unclear;
- not usable: a longer run crashed inside `Llama.eval` (`ValueError: could not broadcast input array`; the scores
  buffer fills to `n_ctx` once a draft model is set). Not enabled in burnish.

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
- **KV cache is not the difference.** At `n_ctx=8192` with a Q8_0 cache, gemma4:26b-a4b uses 935 MiB of
  KV cache and qwen3.8:27b about 0.3 GB. Qwen3.8 also keeps about 150 MB of fixed f32 recurrent state, which
  `type_k`/`type_v` do not quantize. For gemma4:26b-a4b, llama-cpp-python 0.3.36 sizes the cache of all 25
  sliding-window layers for the full context (850 MiB), although the window is only 1024 tokens. The 5
  global layers take 85 MiB.

## Smaller gemma4:26b-a4b quants (RCO-lite)

Goal: fit gemma4:26b-a4b on a 12 GB card. Pipeline: `quant/` (`run_all.sh`).
- Source: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`, converted to bf16 GGUF with llama.cpp `b12020`.
  Its converter needs transformers 5.6.2.
- Only the MoE expert tensors are searched; they are 90% of the weights. Everything else stays as in Google's
  release: Q4_0, with Q6_K token embeddings.
- `ffn_down_exps` rows are 704 wide, so it can only use 32-block types: MXFP4, IQ4_NL or Q4_0.
- The sweep swaps one 3-layer group at a time to each candidate type and measures the KL divergence against
  bf16. The data is wikitext-2: imatrix from 200 train chunks, KLD on 40 test chunks of 512 tokens.
- A knapsack then picks the type mix with the lowest summed KL divergence under each size budget.

| file | size | expert types (gate_up / down) | KLD vs bf16 | same top token | CUDA buffers at n_ctx 8192 |
|---|---|---|---|---|---|
| Google QAT Q4_0 | 13.45 GiB | Q4_0 / Q4_0 | 0.353 | 70.3% | — (needs a 16 GB card) |
| ours: Q4_0 with imatrix | 13.45 GiB | Q4_0 / Q4_0 | **0.208** | **78.3%** | — |
| RCO-lite 11 GiB | 10.99 GiB | mostly IQ3_XXS, IQ3_S at both ends / IQ4_NL, Q4_0, MXFP4 | 0.328 (pred. 0.300) | 71.1% | 12706 MiB |
| RCO-lite 10.5 GiB | 10.48 GiB | IQ3_XXS and IQ2_S / mostly MXFP4 | 0.385 (pred. 0.356) | 69.2% | 12185 MiB |
| RCO-lite 10 GiB | 9.97 GiB | mostly IQ2_S, IQ3_XXS in layers 0–5 / MXFP4 | 0.501 (pred. 0.483) | 66.2% | 11663 MiB |

Raw wikitext is far from what this instruct model is built for: even bf16 has a perplexity of 551 on it. That
inflates every KLD above, so compare the KLD values with each other only. Ours and Google's Q4_0 files come
from the same checkpoint and differ only in the imatrix-weighted rounding. Part of our advantage may come from
calibrating and measuring on the same kind of text.

Burnish eval (104 sentences, all files measured in one session on RTX 4090s):

| file | hybrid exact | all-mode exact | all-mode over-correction | hybrid p50 | all-mode p95 |
|---|---|---|---|---|---|
| Google QAT Q4_0 | 99.0% | 79.8% | 17.3% | 0.16 s | 0.39 s |
| ours: Q4_0 with imatrix | 99.0% | 78.8% | 17.3% | 0.13 s | 0.37 s |
| RCO-lite 11 GiB | 97.1% | 79.8% | 16.3% | 0.13 s | 0.38 s |
| RCO-lite 10.5 GiB | 98.1% | 79.8% | 17.3% | 0.13 s | 0.37 s |
| RCO-lite 10 GiB | 99.0% | 77.9% | 19.2% | 0.12 s | 0.37 s |

Findings:
- **burnish can't tell these files apart.** Every miss is a defensible alternative to the reference:
  - `team have` → `team has`, which all five make
  - `don't need no help` → `don't need help`, where the reference has `any help`
  - `lgtm` → `LGTM`

  The 97–99% spread therefore says nothing about quant quality. The 10 GiB file, with the worst KLD, ties
  Google's file.
- **Fit on a 12 GB card (12288 MiB).** Add ~300–500 MiB of CUDA context to the buffers above. Only the 10 GiB
  file comes close, at about 12.0–12.2 GiB in total. Not verified: llama-cpp-python's `swa_full=False`
  should shrink the sliding-window KV cache from 850 MiB to roughly the window size, about 150 MiB. That
  would bring the 10 GiB file to ~11.3 GiB and the 10.5 GiB file to ~11.8 GiB, including the context.
- **The allocator's predictions run 4–9% low.** Swapping a single 3-layer group moves the KLD by only about
  0.002–0.05, and not in order of bits: IQ4_XS was worse than IQ3_S for layers 0–2.
- **Sweep cost.** About 1.5 hours across two GPUs, mostly the Python assembler writing an 11.5 GB file for
  every variant at about 75 MB/s. The KL measurement itself takes about 20 s.

### Calibration on burnish-style prompts

`quant/run_fix.sh` rebuilds the Q4_0 imatrix from burnish's own kind of input, on the CPU. The calibration set has
483 chat-format fix/polish prompts, 110k tokens. Each one renders Gemma 4's template with burnish's real system
prompts and has the corrected text as the model turn. The texts are wikitext-2 train sentences with synthetic
errors: agreement and homophone swaps, typos, articles, commas, apostrophes and case. The code checks they are
disjoint from `sentences.tsv` and `polish_cases.jsonl`. The KLD is measured on the model turns of 60 held-out
prompts from the same generator (2287 tokens), through llama-cpp-python, because `llama-perplexity` can't parse
the chat tokens.

| file | KLD vs bf16, fix prompts | p99 | same top token | all-mode exact | all-mode over-correction | polish cases (all) |
|---|---|---|---|---|---|---|
| Google QAT Q4_0 | 0.00138 | 0.0223 | 99.78% | 79.8% | 17.3% | 98.9% |
| Q4_0, wikitext imatrix | **0.00088** | **0.0155** | 99.78% | 78.8% | 17.3% | 98.9% |
| Q4_0, fix-prompt imatrix | 0.00098 | 0.0229 | **99.87%** | **82.7%** | **15.4%** | **100%** |

All three score 99.0% hybrid and 99.0% LLM-only exact. The burnish eval ran on GPU 3 while the CPU did the KLD
runs, so its latencies are left out.

Findings:
- **The fix-prompt imatrix doesn't beat the wikitext one on KLD**, even on prompts from its own generator. Both
  imatrix files beat Google's on the mean. The fix-prompt file has a worse tail (p99 as high as Google's) and
  2 more matching top tokens out of 2287.
- **In mode `all`, the new file rewrites less in the polish step.** It keeps `config` and `docs` where the other
  two expand them, and gives `They're` instead of `They are`. It is 3–4 sentences out of 104, within noise.
- The calibration texts are encyclopedic, not chat or email. A corpus closer to burnish's real input might do
  better; this one doesn't justify switching the default `gemma4:26b-a4b-imat` file.
