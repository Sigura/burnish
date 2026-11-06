import argparse
import csv
import json
import statistics
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from burnish.config import load_config
from burnish.engine import LlamaEngine
from burnish.languagetool import LanguageToolClient, LanguageToolError
from burnish.models import ModelStore
from burnish.pipeline import Pipeline, char_ratio
from burnish.protect import protected_spans
from burnish.selector import ModelSelector, query_free_vram_mb, query_process_vram_mb, selection_from_tier

ROOT = Path(__file__).parent
GPU_TARGET_SECONDS = 3.0
CPU_TARGET_SECONDS = 8.0


def word_distance(a: str, b: str) -> int:
    x, y = a.split(), b.split()
    previous = list(range(len(y) + 1))
    for i, word in enumerate(x, 1):
        current = [i]
        for j, other in enumerate(y, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1]


def score(rows: list[tuple[str, str]], outputs: list[str]) -> dict:
    exact = over = spans_kept = spans_total = 0
    similarity = 0.0
    for (source, expected), output in zip(rows, outputs):
        exact += output == expected
        similarity += char_ratio(output, expected)
        # More words touched than the reference correction needed counts as over-correction.
        over += output != expected and word_distance(source, output) > word_distance(source, expected)
        for start, end in protected_spans(source):
            spans_total += 1
            spans_kept += source[start:end] in output
    return {
        "exact": exact / len(rows),
        "similarity": similarity / len(rows),
        "over_correction": over / len(rows),
        "spans_kept": spans_kept / spans_total if spans_total else 1.0,
    }


def run_variant(pipeline, rows, selection, use_llm, use_lt, mode="fix") -> dict:
    outputs, seconds, fallbacks = [], [], 0
    for source, _ in rows:
        started = time.perf_counter()
        result = pipeline.refine(source, mode, selection=selection, use_llm=use_llm, use_lt=use_lt)
        seconds.append(time.perf_counter() - started)
        outputs.append(result.text)
        fallbacks += any(i["type"].endswith("_guard") for i in result.issues)
    ordered = sorted(seconds)
    return {
        **score(rows, outputs),
        "p50": statistics.median(seconds),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "guard_rejections": fallbacks,
        "outputs": outputs,
    }


def print_row(name: str, m: dict, target: float | None) -> None:
    met = "-" if target is None else ("yes" if m["p95"] <= target else "NO")
    print(
        f"| {name:<34} | {m['exact']:>5.0%} | {m['similarity']:>5.3f} | {m['over_correction']:>5.0%} "
        f"| {m['spans_kept']:>5.0%} | {m['guard_rejections']:>3} | {m['p50']:>5.2f} | {m['p95']:>5.2f} | {met:>3} |"
    )


def evaluate_candidate(pipeline, llm, rows, tier, model, on_gpu, results) -> dict:
    selection = replace(selection_from_tier(tier, on_gpu), model=model, keep_alive_seconds=600)
    llm.unload()
    started = time.perf_counter()
    pipeline.refine(rows[0][0], "fix", selection=selection, use_lt=False)
    cold = time.perf_counter() - started
    vram_mb = query_process_vram_mb()

    device = "gpu" if on_gpu else "cpu"
    target = GPU_TARGET_SECONDS if on_gpu else CPU_TARGET_SECONDS
    hybrid = run_variant(pipeline, rows, selection, True, True)
    llm_only = run_variant(pipeline, rows, selection, True, False)
    # `all` is the default mode, so latency targets apply to it. Its output goes beyond the
    # minimal references, which is why the quality columns come from fix mode.
    all_mode = run_variant(pipeline, rows, selection, True, True, mode="all")
    hybrid["p95_all"] = all_mode["p95"]
    print_row(f"{model} {device} hybrid", hybrid, target)
    print_row(f"{model} {device} llm-only", llm_only, target)
    met = "yes" if all_mode["p95"] <= target else "NO"
    print(f"|   mode all: p50 {all_mode['p50']:.2f}s, p95 {all_mode['p95']:.2f}s, target met: {met}")
    print(f"|   cold start {cold:.1f}s, {vram_mb} MiB of VRAM in use")
    llm.unload()
    results[f"{model}/{device}"] = {"cold_seconds": cold, "vram_mb": vram_mb, "hybrid": hybrid, "llm_only": llm_only, "all_mode": all_mode}
    return hybrid


def recommend(label: str, measured: dict[str, dict], target: float) -> None:
    fast = {name: m for name, m in measured.items() if m["p95_all"] <= target}
    if not fast:
        print(f"{label}: no candidate meets {target:.0f}s at p95 in mode all")
        return
    best = max(fast, key=lambda name: (fast[name]["exact"], -fast[name]["over_correction"], -fast[name]["p50"]))
    print(f"{label}: model = \"{best}\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models and the hybrid pipeline.")
    parser.add_argument("--limit", type=int, help="use only the first N sentences")
    parser.add_argument("--no-gpu", action="store_true", help="skip the GPU tiers")
    parser.add_argument("--no-cpu", action="store_true", help="skip the CPU tier")
    args = parser.parse_args()

    with open(ROOT / "sentences.tsv", newline="", encoding="utf-8") as f:
        rows = [(r["input"], r["expected"]) for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    rows = rows[: args.limit]

    cfg = load_config()
    llm = LlamaEngine(cfg.engine, ModelStore(cfg))
    lt = LanguageToolClient(cfg.languagetool)
    try:
        lt.check("ok")
    except LanguageToolError as e:
        raise SystemExit(f"{e}\nStart it with: systemctl --user start burnish-languagetool")
    pipeline = Pipeline(cfg, llm, lt, ModelSelector(cfg.selection, query_process_vram_mb))
    installed = llm.installed()

    results: dict[str, dict] = {}
    print(f"{len(rows)} sentences. Targets at p95: GPU {GPU_TARGET_SECONDS:.0f}s, CPU {CPU_TARGET_SECONDS:.0f}s.\n")
    print("| variant                            | exact |  sim  | over  | spans | grd |  p50  |  p95  | met |")
    print("|---|---|---|---|---|---|---|---|---|")
    lt_only = run_variant(pipeline, rows, None, False, True)
    print_row("languagetool only", lt_only, None)
    results["languagetool"] = lt_only

    picks: list[tuple[str, dict, float]] = []

    def run_tier(label, tier, on_gpu, target):
        measured = {}
        for model in tier.candidates:
            if model not in installed:
                print(f"|   skipped {model}: not downloaded (burnish-models pull {model})")
                continue
            measured[model] = evaluate_candidate(pipeline, llm, rows, tier, model, on_gpu, results)
        picks.append((label, measured, target))

    if not args.no_gpu and not llm.supports_gpu():
        print("|   skipped GPU tiers: llama.cpp was built without GPU support")
    elif not args.no_gpu:
        for tier in cfg.selection.gpu_tiers:
            free = query_free_vram_mb()
            available = None if free is None else free + query_process_vram_mb()
            if available is None or available < tier.min_free_vram_mb:
                print(f"|   skipped GPU tier >= {tier.min_free_vram_mb} MiB: {available} MiB free")
                continue
            run_tier(f"GPU tier >= {tier.min_free_vram_mb} MiB", tier, True, GPU_TARGET_SECONDS)
    if not args.no_cpu:
        run_tier("CPU tier", cfg.selection.cpu, False, CPU_TARGET_SECONDS)

    print("\nSuggested defaults (best exact match within the latency target):")
    for label, measured, target in picks:
        recommend(label, measured, target)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps({"inputs": rows, "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull outputs: {out_path}")


if __name__ == "__main__":
    main()
