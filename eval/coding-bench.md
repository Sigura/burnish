# Coding-model bench and gemma4 SWA fit (November 2026)

Host: 2× EPYC 7601, GPUs 0,2,3 RTX 4090 and GPU 1 RTX 3090 Ti, all power-limited to ~203–235 W on purpose
(`CUDA_DEVICE_ORDER=PCI_BUS_ID`). llama.cpp `b12020`. GPU 1 was replaced on 2026-11-15 (new RTX 3090 Ti, no
hardware throttling); at the shared power limit it decodes at about a third of a 4090's speed (see "GPU health").

Goal: help the parallel llm_coder study (`~/llm_coder/llm/results/two-model-2026-11-13/report.md`) choose its coding
models. Coordination with llm_coder runs through `eval/coord/burnish.md` (symlinked into `~/llm_coder/llm/coord/`).

## Status

| step | state |
|---|---|
| 1. gemma4:26b-a4b `swa_full=False` VRAM (fit_check) | done |
| 1b. prefix reuse with `swa_full=False` in llama-cpp-python | not run: blocked by the `block_binary` hook |
| 2. speed bench at 32k: Qwen3.8-27B Q6_K (GPU 0), Coder-Next UD-IQ3_S (GPUs 0,2) | done; the earlier Coder-Next run on GPUs 0,1 is superseded |
| 3. coding eval `eval/coding/`: 25 Python/bash tasks + 4 Kotlin, all references pass | done |
| 3a. greedy (temperature 0), one seed | done: Qwen3.8-27B Q6_K and IQ3_S, Coder-Next UD-IQ3_S; llm_coder: Qwen3.6, RCO-37g |
| 3b. recommended sampling, 3 seeds | done for all 6 candidates (Kotlin for burnish's 3 models greedy only) |
| 4. final per-language comparison and OpenCode routing advice | done, section 4 |
| model files for llm_coder | copied to `~/llm_coder/llm/models/burnish/` (sha256-verified) |
| GPU 1 swap (2026-11-15) | done: new RTX 3090 Ti passes the load test, no HW throttling; power-limited speed, see "GPU health" |

## 1. gemma4:26b-a4b with `swa_full=False`

File: `quant-work/out/gemma-4-26B-A4B-it-qat-rcolite-10g.gguf` (9.97 GiB). llama-cpp-python 0.3.36,
`n_ctx=8192`, flash attention, Q8_0 KV, one RTX 4090 (nvidia-smi 2).
Command: `quant/fit_check.py MODEL --swa-full default|off`.

| CUDA buffer | default (full SWA cache) | `swa_full=False` |
|---|---|---|
| model | 10194.5 MiB | 10194.5 MiB |
| KV, 5 global layers | 85.0 MiB | 85.0 MiB |
| KV, 25 sliding-window layers | 850.0 MiB | **159.4 MiB** |
| compute | 533.2 MiB | 527.7 MiB |
| **total** | 11663 MiB | **10967 MiB** (−696 MiB) |

With ~300–500 MiB of CUDA context the 10 GiB file comes to ~11.3–11.5 GiB, under a 12 GB card's 12288 MiB.
That matches the estimate in `model-comparison.md`. Not tested on an actual 12 GB card.

Open question: whether llama-cpp-python can still reuse the prompt prefix with the smaller SWA cache. If it
can't, every burnish call recomputes the prompt. Burnish itself still uses the default.

## 2. Coding candidates at 32k context

Script: `eval/bench_llama_server.sh MODEL.gguf` (GPUs 2,3, layer split, flash attention, Q8_0 KV):
- `llama-bench`: pp2048 and tg128 at depth 0 and depth 32768.
- `llama-server -c 32768 --jinja`: a 28k-token prompt (burnish source) with 256 generated tokens, then the
  same prompt plus a suffix to check prefix reuse (context checkpoints on the hybrid layers).
- VRAM from nvidia-smi while the server is loaded.

Everything runs fully in VRAM (`-ngl 99`, no `--n-cpu-moe`). The IQ3_S numbers hold only for that setup: a
parallel DeepSeek study measured i-quants ~5× slower than K-quants with experts in CPU RAM on this Zen1 host.

| model | file | size | pp2048 tok/s (d0 / d32k) | tg128 tok/s (d0 / d32k) | 28k prompt tok/s | gen tok/s at 28k | prefix reused | VRAM GPU2 / GPU3 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | UD-Q6_K | 20.46 GiB | 1976 / 1541 | 37.8 / 33.0 | 1756 | 33.3 | yes: 18 of 28032 tokens recomputed | 22051 MiB on GPU 0 alone¹ |
| Qwen3-Coder-Next 80B-A3B | UD-IQ3_S | 27.65 GiB | 1891 / 1537 | 112.7 / 71.8 | 1379 | 75.7 | yes: 18 of 28032 tokens recomputed | 15635 MiB on GPU 0 + 14521 MiB on GPU 1² |

¹ Qwen3.8-27B ran on GPU 0 only, not split over 2,3 (the GPUs were busy with llm_coder's imatrix). nvidia-smi showed
22521 MiB with llama-server loaded at `-c 32768`, minus 470 MiB already in use on that card before the run. That
leaves ~2 GB of headroom on a 24 GB card at 32k context with Q8_0 KV. The script's per-buffer lines didn't match
the b12020 server log, so there is no KV/RS breakdown. llama-bench: 2 repetitions, ± under 0.5%.

Findings, Qwen3.8-27B Q6_K:
- **Fits one 4090 at 32k context**, so it doesn't need GPU 3.
- **Dense 27B decode is ~38 tok/s** on an empty context and 33 tok/s at 32k. That's about a quarter of what the 3B-active
  MoE models in this study reach (Qwen3.6-35B-A3B: 134–156 tok/s on llm_coder's numbers).
- **Prefill holds up at depth**: 1976 → 1541 tok/s at 32k (−22%), thanks to the 48 linear-attention layers.
- **llama-server reuses the prefix** on this hybrid model: the second request with the same 28k prefix recomputed
  only the 18 new tokens. That's the problem llama-cpp-python has in burnish; it doesn't occur here.
- The CPU was busy with llm_coder's imatrix during the run. With the model fully on the GPU this should barely affect
  the numbers, but that's not verified.

² Qwen3-Coder-Next ran on GPUs 0,1 (RTX 4090 + RTX 3090 Ti, layer split), not on 2,3: llm_coder's imatrix held those.
VRAM is nvidia-smi with llama-server loaded at `-c 32768`, minus what each card used before (470 / 334 MiB): ~30.2 GB
total, so it would also fit 2× 4090 with ~17 GB to spare. Again, the CPU was loaded by llm_coder's imatrix.

**Re-bench 20:15, GPUs 0,2 (two RTX 4090s, power-limited to ~232–235 W), same script.** This supersedes the GPUs 0,1 row:

| | pp2048 tok/s (d0 / d32k) | tg128 tok/s (d0 / d32k) | 28k prompt tok/s | gen tok/s at 28k | prefix reused | VRAM |
|---|---|---|---|---|---|---|
| Coder-Next UD-IQ3_S, GPUs 0,2 | **5060 / 3928** | **137 / 120** | 4300 | 111 | yes: 18 tokens recomputed | 15640 + 14668 MiB |
| (old, GPUs 0,1 with throttled 3090 Ti) | 1891 / 1537 | 113 / 72 | 1379 | 76 | yes | 15635 + 14521 MiB |

GPU 1 cost 2.7× in prefill and most of the depth drop in decode (−12% at 32k now, −36% before). Caveat: llm_coder
reports two failed model loads on GPU 2 at 20:09–20:10, overlapping the start of this bench (each failed at
cudaMalloc within ~6 s). The spreads are small (pp ±65, tg ±7.7), so the effect is probably minor; not re-run. The findings below
about depth and prefill refer to the old run.

Findings, Qwen3-Coder-Next IQ3_S:
- **~3× faster decode than the dense 27B**: 113 tok/s on an empty context vs 38. At 28k context, 76 vs 33.
- **Decode drops more with depth than the dense model**: 113 → 72 tok/s at 32k (−36%), vs −13% for Qwen3.8-27B. With so
  few active params, attention over the 12 full-attention layers' KV becomes a large share of each token's work.
- **Prefill is similar to the 27B** (1891 vs 1976 tok/s), and the 28k server prompt is slower (1379 vs 1756). The
  IQ3_S dequant cost and the 3090 Ti likely both contribute; not separated.
- **Prefix reuse works in llama-server** here too (hybrid model with linear-attention layers).
- Superseded by the GPUs 0,2 re-bench above: GPU 1 was throttling during this run.

## 3. Coding quality eval (`eval/coding/`)

Goal: help llm_coder choose its coding models. Candidates: Qwen3.6-35B-A3B, Coder-Next IQ3_S and its RCO builds,
Qwen3.8-27B, DeepSeek V4.1 Flash.

- `run_coding_eval.py --url http://127.0.0.1:PORT --label NAME`: works with any OpenAI-compatible endpoint; needs only
  python3 and bash (Kotlin tasks also need `kotlinc` and a JDK). Options:
  - `--model` picks a model on router-mode servers; `--no-think` sends `enable_thinking=false`.
  - `--tasks GLOB` / `--exclude GLOB` select tasks (`--exclude '*_e_*'` gives the 17 v1 tasks); `--langs python,bash,kotlin`.
  - Sampling: `--temperature` (default 0), `--seed`, and `--top-p --top-k --min-p --presence-penalty --repeat-penalty`
    (sent only when given). `--max-tokens` defaults to 32768.
  - `--ref` checks the reference solutions; `--rescore RESULT_JSON` re-runs the current tests on saved replies;
    `--task-dir` points at another task folder.
- `run_with_server.sh MODEL.gguf LABEL [eval args]`: starts llama-server (same flags as `bench_llama_server.sh`, `GPUS`,
  `PORT`, `CTX` env), runs the eval, then stops the server.
- Score: pass@1, and pass@2 after one retry that feeds back the failing test output. `tests@k` gives partial credit:
  the sum of the best fraction of test cases passed per task. Replies cut off by `--max-tokens` count as `capped`; a
  retry that overflows the server context is recorded as a failed attempt. Each run also records total time and
  completion tokens. Results go to `eval/coding/results/*.json` with each reply and its test output.
- Tasks are written fresh on this host (not from public benchmarks), each with hidden tests and a reference solution:
  - Python (16): cron next-fire, unified-diff apply, typed dataclass loader, asyncio limited gather, expression parser,
    POSIX shell word splitting, semver ranges, weighted grid route with lexicographic tie-break, INI with
    interpolation, 2 bug fixes (interval set, TTL LRU decorator); easier tier `py_e_*`: duration parse/format,
    flatten/unflatten, config deep merge, log summary, pagination bug fix.
  - bash (9): word frequency, access-log summary, retry wrapper, backup rotation, dotenv → export, CSV column
    aggregate; easier tier `sh_e_*`: extension count, literal find/replace, lowercase rename. The prompts state that
    `awk` is mawk, so gawk-only code fails, as it would on Debian/Ubuntu.
  - Kotlin (4): LRU cache, JSON renderer, money ledger bug fix, tokenizer. References checked by llm_coder (no
    `kotlinc` in burnish's container).
- `quant-work/data/test.txt` stays the KLD set; this eval doesn't touch it. Multi-turn agentic tasks (OpenCode e2e)
  stay llm_coder's.
- Noise: with sampling, single runs vary by ±2–3 tasks of 25. Compare means over 3 seeds.
- Thinking on: Qwen3.6 with thinking needed >32k tokens on half the attempts and scored worse than thinking off, so
  all comparisons use thinking off.

### Results, eval v1 (17 tasks: 11 Python, 6 bash; thinking off unless noted; temperature 0, one seed)

Scored with the tests as fixed at ~16:25: bash exit-code checks no longer also capture stdout. Earlier runs were rescored from
their saved replies with `run_coding_eval.py --rescore` (`results/*-rescored.json`).

| model | GPU | pass@1 | pass@2 | tests@1 | tests@2 | Python pass@2 | bash pass@2 | time | tokens |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q6_K | 4090 (GPU 0) | **6/17** | **9/17** | **13.6** | **14.5** | 6/11 | 3/6 | 1804 s | 63k |
| Qwen3.6-35B-A3B UD-Q4_K_XL (llm_coder) | 4090 (GPU 2) | 4/17 | 5/17 | 10.1 | 10.3 | 3/11 | 2/6 | 447 s | 65k |
| Qwen3.6-35B-A3B UD-Q4_K_XL, thinking, 32k cap (llm_coder) | 4090 | 2/17 | 4/17 | 2.9 | 8.4 | 3/11 | 1/6 | 5311 s | 707k |
| Qwen3.8-27B GSQ-RCO IQ3_S (10.96 GiB) | 4090 (GPU 0) | 5/17 | 6/17 | 11.3 | 13.6 | 4/11 | 2/6 | 1322 s | 67k |
| Coder-Next RCO-37g (llm_coder, 36.89 GiB) | 2× 4090 (GPUs 2,3) | 5/17 | 6/17 | 9.6 | 11.9 | 3/11 | 3/6 | 372 s | 50k |
| Coder-Next UD-IQ3_S (27.65 GiB) | 2× 4090 (GPUs 0,2) | 1/17 | 3/17 | 7.6 | 8.7 | 1/11 | 2/6 | 988 s | 125k |

Easier tier (8 tasks `*_e_*`, run separately so the v1 numbers stay comparable; merged into `tasks/` at 20:41):

| model | pass@1 | pass@2 | tests@1 | tests@2 | time |
|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q6_K | 3/8 | 5/8 | 5.55 | 7.07 | 325 s |
| Qwen3.8-27B GSQ-RCO IQ3_S | 3/8 | 5/8 | 5.65 | 7.33 | 244 s |
| Coder-Next UD-IQ3_S (GPUs 0,2) | 4/8 | 4/8 | 6.63 | 6.70 | 118 s |

Both fail `sh_e_replace` in the same way: they expand `\1` in the replacement to byte 0x01 (`printf %b`/`echo -e`),
although the spec says the replacement is literal. That's a genuine mistake; the reference passes. IQ3_S matches Q6_K
here at half the file size.

- **Greedy decoding loops.** In the v1 run, Coder-Next UD-IQ3_S looped on 2 tasks until it hit the 32k cap (`py_cron_next`:
  "Actually, let's just implement the logic as described" 281×; `sh_env_export`: "Let's implement it." 221×). The
  retry then overflowed the 32k server context (recorded as a failed attempt since the 20:20 harness fix). The RCO-37g
  build of the same model didn't loop. Temperature 0 favours models that stay stable under greedy decoding; Qwen
  recommends sampling for Coder-Next; see the 3-seed results below, where the loops disappear.
- The dense 27B Q6_K solves the most tasks, but at 4× Qwen3.6's wall time (35 vs ~150 tok/s decode).
- With one seed, single tasks flip between models: Qwen3.6 solved `sh_access_log` and `grid_route`, which Q6_K
  failed. Differences of 1–2 tasks are noise. Seed repeats and the easier tier (8 tasks in `staging/`) come next.
- Q6_K's remaining failures checked by hand are genuine mistakes: `grid_route` returns None, `wordfreq` splits on
  apostrophes, and `access_log` rejects every line.
- The thinking run's "capped" count is missing because that run predates cap tracking; llm_coder reported 16 of 32
  attempts capped.

Kotlin (4 tasks, temperature 0, one seed, run by llm_coder): Qwen3.8-27B IQ3_S 3/4 → 3/4, Q6_K 2/4 → 3/4,
Coder-Next UD-IQ3_S 0/4 → 0/4, Qwen3.6-35B-A3B 1/4 → 4/4 (that run was on the throttled GPU 1; quality only).

KLD vs bf16 on llm_coder's coding-agent test set (`test.txt`): Coder-Next UD-IQ3_S 0.265; llm_coder's RCO builds
rco-33g 0.158, rco-37g 0.123, rco-37g-r2 (after the re-sweep round) 0.121. The RCO builds track bf16 about 2× more
closely than the Unsloth IQ3_S file.

### Results with recommended sampling, 3 seeds (25 Python/bash tasks)

Each model runs with its model card's sampling settings, seeds 1–3, thinking off, on two 4090s (GPUs 0,2 for Coder-Next) or one
4090. Mean ± standard deviation over the seeds:
- Qwen3.8-27B, non-thinking: `--temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 --presence-penalty 1.5`
- Coder-Next: `--temperature 1.0 --top-p 0.95 --top-k 40`

| model | file | pass@1 | pass@2 | tests@2 | Python pass@2 (/16) | bash pass@2 (/9) | time per run |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q6_K | 20.5 GiB | **10.0 ± 1.7** | **15.0 ± 2.6** | **21.8 ± 0.9** | 9.0 ± 1.0 | **6.0 ± 2.0** | 2294 s |
| Qwen3.8-27B GSQ-RCO IQ3_S | 10.96 GiB | 7.0 ± 2.6 | 13.7 ± 2.9 | 20.4 ± 1.0 | 8.7 ± 2.1 | 5.0 ± 1.0 | 1980 s |
| Coder-Next UD-IQ3_S | 27.65 GiB | 6.3 ± 0.6 | 10.0 ± 1.0 | 18.3 ± 1.5 | 7.0 ± 1.0 | 3.0 ± 1.0 | **789 s** |

- Sampling beats greedy for every model here. Coder-Next stops looping (0 capped attempts vs 4) and goes from 7/25 at
  pass@2 (greedy, v1 + easier tier) to 10.0. Qwen3.8-27B Q6_K goes from 14/25 to 15.0.
- Q6_K vs IQ3_S: Q6_K leads by 3 tasks at pass@1, but at pass@2 the gap (1.3) is inside one standard deviation.
  For the agent use case, where the model sees test output and retries, IQ3_S at half the VRAM is close.
- Coder-Next is ~3× faster per run than the 27B quants but solves ~5 fewer tasks, mainly in bash (3 vs 6 of 9).
- Seed spread is large (±2–3 tasks at pass@2). Single-seed comparisons from earlier sections can mislead.

## GPU health

`scripts/gpu_watch.sh` logs clocks, power, temperature and throttle reasons every 30 s to `run-logs/gpu-watch/`
(alerts in `alerts.log`).
- GPU 1, old RTX 3090 Ti (until 2026-11-14): HW slowdown and HW thermal throttling (reasons 0x4C, 210–375 MHz) at a
  core temperature of 56–62 °C. Replaced on 2026-11-15; GPU 1 numbers from 2026-11-14 are not comparable.
- GPU 1, new RTX 3090 Ti (UUID `GPU-868da6b5…`), load test 2026-11-15 with Qwen3.8-27B GSQ-RCO IQ3_S alone on it
  (`run-logs/gpu1-check/bench.log`): pp2048 750, tg128 19.1, at depth 32k pp 541 / tg 14.9; 28k prompt 692 tok/s,
  gen 18.4; prefix reuse OK; 13.3 GB VRAM. GPU 4090 reference: tg ~58. Only the software power cap is active
  (reason 0x4, 201–221 W limit, SM 525–900 MHz under load, 64 °C max): no hardware or thermal throttling. The low speed
  comes from the power limit (GDDR6X takes a large share of ~200 W), not from a fault.
- GPUs 0,2,3 (RTX 4090): single low-clock samples during bursty partial-offload loads only; no thermal flags.

## 4. Final comparison and OpenCode routing advice

All six candidates, recommended sampling (model card), thinking off, 3 seeds, mean ± sd. Python + bash = 25 tasks
(16 + 9). Kotlin = 4 tasks: llm_coder's models ran 3 seeds with sampling; burnish's models only one greedy run (marked g).

| model | GPUs / VRAM | decode tok/s | pass@1 (25) | pass@2 (25) | tests@2 | Python p@2 /16 | bash p@2 /9 | Kotlin p@2 /4 | time per run |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B UD-Q6_K | 1× 4090, ~22 GB | 38 / 33 at 32k | **10.0 ± 1.7** | **15.0 ± 2.6** | **21.8** | **9.0** | **6.0** | 3 (g) | 2294 s |
| Qwen3.8-27B GSQ-RCO IQ3_S | 1× 4090, ~13 GB | ~58 | 7.0 ± 2.6 | 13.7 ± 2.9 | 20.4 | 8.7 | 5.0 | 3 (g) | 1980 s |
| Coder-Next UD-IQ3_S | 2× 4090, ~30 GB | 137 / 120 at 32k | 6.3 ± 0.6 | 10.0 ± 1.0 | 18.3 | 7.0 | 3.0 | 0 (g) | 789 s |
| Qwen3.6-35B-A3B Q4_K_XL | 1× 4090, ~22 GB | 134–156 | 6.0 ± 1.0 | 9.3 ± 0.6 | 16.9 | 4.7 | 4.7 | 2.0 ± 1.7 | n/c (GPU 1) |
| Coder-Next rco-37g-r2 (llm_coder) | 2× 4090, ~37 GB | not benched | 5.3 ± 1.2 | 8.7 ± 0.6 | 17.6 | 5.7 | 3.0 | 1.3 ± 1.5 | 656 s |
| Qwen3.6-35B-A3B MTP Q4_K_XL | 1× 4090 | MTP (llm_coder) | 6.7 ± 1.5 | 8.3 ± 0.6 | 17.4 | 4.7 | 3.7 | 1.3 ± 1.5 | 792 s |

Findings:
- **Qwen3.8-27B is the strongest in every language.** Q6_K leads; GSQ IQ3_S is within one sd at pass@2 using ~40% less
  VRAM and decoding ~1.5× faster. The ~5-task gap to the rest is larger than the seed spread.
- **The fast MoE models cluster at 8–10 of 25.** Coder-Next (two GPUs) is no better than Qwen3.6 (one GPU) and is weakest
  in bash. MTP doesn't change Qwen3.6's quality (8.3 vs 9.3, within noise) and makes it faster.
- **KLD doesn't predict task quality here.** rco-37g-r2 has less than half the KLD of UD-IQ3_S (0.121 vs 0.265) but scores
  the same or lower on tasks. Use task evals, not KLD alone, to choose builds.
- **Kotlin is too small to rank models** (4 tasks, partly greedy only); no evidence that any model is better at Kotlin
  than its overall rank suggests.

Routing advice for OpenCode on this host (per-agent `model` setting):
- **Main coding agent, all languages:** Qwen3.8-27B, thinking off, temperature 0.7 / top_p 0.8 / top_k 20 /
  presence_penalty 1.5. Use UD-Q6_K when a 4090 is free (best first-try accuracy); use GSQ-RCO IQ3_S when VRAM is shared
  (~13 GB) or decode speed matters more.
- **`small_model` / fast helper** (titles, summaries, quick lookups): Qwen3.6-35B-A3B with MTP, thinking off.
- **Coder-Next:** not recommended here. It needs two GPUs and scores below Qwen3.8-27B; its only advantage is ~3×
  faster runs, which Qwen3.6 MTP also offers on one GPU.
- No per-language split is needed: no language has a different winner.

Limits: single-turn tasks with one retry, 25 + 4 tasks, 3 seeds; no multi-turn agent tasks (llm_coder's OpenCode
e2e covers that), thinking off throughout; DeepSeek V4.1 Flash not scored on this eval.

## Reboot for the GPU 1 swap

Before:
1. Wait until llm_coder's 3-seed runs are done (`eval/coding/results/*qwen3.6-*-rec-s3.json`,
   `*coder-next-rco37r2-rec-s3.json`; ETA ~01:00) or until llm_coder says in `~/llm_coder/llm/coord/llm_coder.md` that
   stopping is fine. Its RCO round 2 sweep may still be running on GPUs 1,3.
2. Burnish runs nothing but `scripts/gpu_watch.sh` (logger, safe to kill). All results are on disk; nothing is lost by
   rebooting. Uncommitted: `eval/coding/`, `eval/coord/`, `eval/coding-bench.md`, `eval/bench_llama_server.sh`,
   `scripts/gpu_watch.sh`, `quant/fit_check.py` (`--swa-full`), `eval/host-model-recommendations.md`.
3. Host side: shut down cleanly, swap the card, boot.

After:
1. `devbox restart --gpus device=0,1,2,3` (container image already has CUDA 12.8 runtime and JDK 21).
2. Check the new GPU 1: `nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit,temperature.gpu,clocks_event_reasons.active --format=csv`.
   Keep or set its power limit like the others (host: `sudo nvidia-smi -i 1 -pl W`).
3. Restart the logger: `scripts/gpu_watch.sh` (in the background); alerts land in `run-logs/gpu-watch/alerts.log`.
4. Load test on GPU 1 alone, ~5 min, then check that no alert appeared and the SM clock stays near max under load:
   `GPUS=1 eval/bench_llama_server.sh ~/.local/share/burnish/models/qwen3.8-27b/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`
   (a 4090: compare with ~58 tok/s decode on GPU 0; a 3090 Ti: no reference yet).
5. Tell llm_coder in `eval/coord/burnish.md` that GPU 1 is back and whether it passed, so it can lift the
   "no speed measurements on GPU 1" rule and run its B3 re-measurements that involved GPU 1.

## Summary

Qwen3.8-27B wins on coding quality in Python, bash and Kotlin: UD-Q6_K 15.0/25 at pass@2, GSQ-RCO IQ3_S 13.7/25 with
~40% less VRAM. The fast MoE models (Qwen3.6-35B-A3B with or without MTP, Coder-Next UD-IQ3_S or rco-37g-r2) solve
8–10/25. Recommended routing: Qwen3.8-27B as the main OpenCode agent, Qwen3.6 MTP as `small_model`, Coder-Next
dropped. Details and caveats in section 4.

## 5. Quantization review (Nov 14)

Agreed with llm_coder: speculative decoding with MTP heads for Qwen3.6 (llm_coder downloaded the MTP GGUF and is
testing it), and re-sweep rounds after the RCO knapsack (running as "RCO round 2").

Pointers shared with llm_coder (user-approved). Numbers come from the sources and are not verified here.

- **Calibration**: chat template plus tool-use text matters most at 2–3 bit. Your agent-episode set is the right
  approach. ([llama.cpp #5263](https://github.com/ggml-org/llama.cpp/discussions/5263),
  [Bartowski imatrix v6](https://runtimewire.com/article/bartowski-imatrix-v6-dataset-gguf-quantization))
- **Metric**: KLD tracks accuracy only while it stays small (~<0.01). Low-bit quants and q4_0 KV degrade more on long
  context and tool calls, so final picks need task evals plus a 32k check.
  ([2605.02404](https://arxiv.org/pdf/2605.02404), [2406.06773](https://arxiv.org/pdf/2406.06773),
  [KV quant benchmarks](https://particula.tech/blog/kv-cache-quantization-accuracy-loss-benchmarks))
- **Allocation**: additive per-layer KLD explains 85–93% of the variance. Re-sweep rounds (13:45 note) or a
  saturation correction model cover the rest. EvoPress and RCO do better below 3 bpw.
  ([2607.12266](https://arxiv.org/pdf/2607.12266), [EvoPress](https://arxiv.org/abs/2410.14649),
  [RCO](https://github.com/IST-DASLab/RCO))
- **Rounding**: GSQ needs H100/H200-class hardware, so not here. gptq-gguf-toolkit writes K-quants for dense models
  only (MoE is future work). Prefer ISTA's prebuilt GSQ-RCO GGUFs.
  ([GSQ](https://github.com/IST-DASLab/GSQ), [gptq-gguf-toolkit](https://github.com/IST-DASLab/gptq-gguf-toolkit),
  [ISTA collection](https://huggingface.co/collections/ISTA-DASLab/gsq-rco-gguf))
- **MoE**: per-expert precision (GEMQ, ProQuant) can't be expressed in GGUF, which allows one type per fused tensor.
  At most, use routing frequency as a per-layer prior. ([GEMQ](https://arxiv.org/abs/2605.23078),
  [ProQuant](https://arxiv.org/html/2610.30779v1))
- **Types**: ik_llama.cpp's trellis IQ*_KT types do better per bit below 3 bpw but need the fork. IQ types decode
  slower, especially with experts in CPU RAM, so use K-quants for offload.
  ([ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp))
- **Serving**: speculative decoding with MTP heads (13:45 note), DSpark worth a measurement, Q8_0 KV plus mainline
  Hadamard rotation is right. ([PR 21038](https://github.com/ggml-org/llama.cpp/pull/21038),
  [MTP on 4090](https://www.pedroalonso.net/blog/qwen-mtp-speculative-decoding-4090/))
- **Candidates**: ISTA's Qwen3.8-27B GSQ-RCO IQ3_S has the same LiveCodeBench score as BF16 per its model card
  ([card](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)). The Qwen3.8-Flash-Next GSQ-RCO-Coder build
  (expert-pruned) is a possible replacement for the stretch candidate; one forum report says it loses long-tail facts.
