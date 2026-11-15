"""RCO-lite: budget-constrained per-tensor quant allocation for gemma4:26b-a4b with stock llama.cpp tools.

Pipeline (each step is a subcommand, run in order):
  data      wikitext-2 train (imatrix calibration) and test (KL divergence)
  corpus    burnish-style calibration and held-out test sets: chat-format fix/polish prompts (burnish venv)
  convert   bf16 QAT checkpoint -> bf16 GGUF
  imatrix   importance matrix from the bf16 GGUF
  kld-base  reference log-probs of the bf16 model
  quantize  one full GGUF per candidate expert type (the tensor database)
  sweep     KL divergence of the base mix with one layer group swapped to each alternative
  allocate  multiple-choice knapsack over the sweep deltas under a size budget
  build     assemble the allocated mix and measure its real KL divergence
  quantize-fix  Q4_0 with the imatrix from `imatrix --corpus fix`
  kld-fix   KL divergence vs bf16 on the model turns of the held-out fix set (burnish venv)

Only the MoE expert tensors are searched; they are 90% of the weights. Everything else keeps the
layout of Google's QAT Q4_0 release (Q4_0, token embeddings Q6_K).
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from urllib.request import urlretrieve

WORK = Path(os.environ.get("WORK", Path.home() / ".local/share/burnish/quant-work"))
LLAMA = WORK / "llama.cpp"
BIN = LLAMA / "build/bin"
PY = WORK / "venv/bin/python"
SRC = WORK / "src/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
BF16 = WORK / "gguf/gemma4-26b-a4b-qat-bf16.gguf"
IMATRIX = WORK / "imatrix.gguf"
IMATRIX_FIX = WORK / "imatrix-fix.gguf"
DATA = WORK / "data"
FIX_CALIB = DATA / "fix.calib.txt"
FIX_TEST = DATA / "fix.test.jsonl"
KLD_BASE = WORK / "kld/bf16.kld"
KLD_FIX_BASE = WORK / "kld/bf16-fix.npy"
EVAL = Path(__file__).resolve().parent.parent / "eval"
DB = WORK / "db"
SWEEP = WORK / "sweep.json"
OUT = WORK / "out"
TMP = WORK / "tmp"

sys.path.insert(0, str(LLAMA / "gguf-py"))
# CUDA orders devices fastest-first by default, so CUDA_VISIBLE_DEVICES=N would not match nvidia-smi's N on this host.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

N_LAYERS = 30
GATE_UP = "ffn_gate_up_exps"
DOWN = "ffn_down_exps"
# ffn_down_exps rows are 704 wide, which rules out every 256-block type (K-quants, i-quants).
OPTIONS = {
    GATE_UP: ["iq2_s", "iq3_xxs", "iq3_s", "iq4_xs", "q4_0"],
    DOWN: ["mxfp4", "iq4_nl", "q4_0"],
}
BASE = {GATE_UP: "iq3_s", DOWN: "q4_0"}
KLD_CHUNKS = 40
CTX = 512


def run(cmd, log=None, env=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    if log is None:
        subprocess.run(list(map(str, cmd)), check=True, env=env)
        return ""
    with open(log, "w") as f:
        subprocess.run(list(map(str, cmd)), check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
    return Path(log).read_text(errors="replace")


def need(*paths):
    for p in paths:
        if not Path(p).exists():
            raise SystemExit(f"missing {p}: run the earlier step first")


def db_file(tensor, qtype):
    other = DOWN if tensor == GATE_UP else GATE_UP
    mix = {tensor: qtype, other: BASE[other]}
    return DB / f"gu-{mix[GATE_UP]}_dn-{mix[DOWN]}.gguf"


def cmd_data(_):
    DATA.mkdir(parents=True, exist_ok=True)
    zpath = DATA / "wikitext-2-raw-v1.zip"
    if not zpath.exists():
        urlretrieve("https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip", zpath)
    with zipfile.ZipFile(zpath) as z:
        for split in ("train", "test"):
            (DATA / f"wiki.{split}.raw").write_bytes(z.read(f"wikitext-2-raw/wiki.{split}.raw"))
    print("data ready:", *sorted(p.name for p in DATA.iterdir()))


def wiki_sentences(lines):
    """Clean sentences of 8-40 words per paragraph, with wikitext's tokenization spaces undone."""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("="):
            continue
        text = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        text = re.sub(r" (?=[,.;:!?%)\]]|'s\b|n't\b)", "", text)
        text = re.sub(r"(?<=[(\[]) ", "", text)
        sents = [s for s in re.split(r"(?<=[.!?])(?<! [A-Z]\.)(?<!Mr\.)(?<!Dr\.)(?<!St\.) (?=[A-Z])", text)
                 if 8 <= len(s.split()) <= 40 and not re.search(r'[";<>@–—]|\(|\)', s) and s[-1] in ".!?"]
        if sents:
            yield sents


SWAPS = {"was": "were", "were": "was", "has": "have", "have": "has", "is": "are", "are": "is", "does": "do",
         "do": "does", "their": "there", "there": "their", "its": "it's", "it's": "its", "than": "then",
         "then": "than", "your": "you're", "a": "an", "an": "a", "to": "too", "who": "whom"}


KINDS = {
    "swap": lambda i, lead, core, tail: core.lower() in SWAPS,
    "typo": lambda i, lead, core, tail: len(core) >= 5 and core.isalpha(),
    "apostrophe": lambda i, lead, core, tail: "'" in core,
    "drop": lambda i, lead, core, tail: i > 0 and not lead and not tail and core.lower() in ("the", "a", "an"),
    "comma": lambda i, lead, core, tail: tail == ",",
    "double": lambda i, lead, core, tail: not lead and not tail and len(core) <= 4,
    "lower": lambda i, lead, core, tail: i == 0 and core[0].isupper(),
}


def corrupt_once(words, rng):
    # Pick the kind first, then a word it applies to, so rare kinds such as agreement swaps are not starved.
    kind = rng.choice(["swap"] * 4 + ["typo"] * 3 + ["apostrophe", "drop", "comma", "double", "lower"])
    parts = [re.match(r"^(\W*)([\w']+)(\W*)$", w) for w in words]
    spots = [i for i, m in enumerate(parts) if m and KINDS[kind](i, *m.groups())]
    if not spots:
        return
    i = rng.choice(spots)
    lead, core, tail = parts[i].groups()
    if kind == "swap":
        new = SWAPS[core.lower()]
        words[i] = lead + (new.capitalize() if core[0].isupper() else new) + tail
    elif kind == "typo":
        j = rng.randrange(1, len(core) - 1)
        core = rng.choice([core[:j] + core[j + 1] + core[j] + core[j + 2:], core[:j] + core[j + 1:],
                           core[:j] + core[j] + core[j:]])
        words[i] = lead + core + tail
    elif kind == "apostrophe":
        words[i] = lead + core.replace("'", "") + tail
    elif kind == "drop":
        del words[i]
    elif kind == "comma":
        words[i] = lead + core
    elif kind == "double":
        words.insert(i, core.lower() if i else core)
    else:
        words[0] = lead + core[0].lower() + core[1:] + tail


def corrupt(text, rng):
    """One to three edits of the kinds in eval/sentences.tsv: agreement, homophones, typos, apostrophes, articles."""
    words = text.split(" ")
    target = rng.choice([1, 1, 2, 2, 3])
    for _ in range(50):
        before = list(words)
        corrupt_once(words, rng)
        if words != before:
            target -= 1
            if not target:
                break
    return " ".join(words)


def eval_texts():
    """Every sentence of the burnish evals, lower-cased, to keep them out of the corpus."""
    texts = set()
    for line in (EVAL / "sentences.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        texts.update(line.split("\t"))
    for line in (EVAL / "polish_cases.jsonl").read_text(encoding="utf-8").splitlines():
        texts.add(json.loads(line)["input"])
    return {" ".join(t.lower().split()) for t in texts if t.strip()}


def cmd_corpus(args):
    """Chat-format fix/polish prompts as burnish sends them, each followed by the corrected text as the model turn.

    Source texts are wikitext-2 train sentences with synthetic errors; the last 20% of paragraphs give the test set.
    """
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    from burnish.pipeline import system_prompt

    need(BF16, DATA / "wiki.train.raw")
    llama = Llama(str(BF16), vocab_only=True, verbose=False)
    special = lambda t: llama.detokenize([t], special=True).decode()
    fmt = Jinja2ChatFormatter(template=llama.metadata["tokenizer.chat_template"],
                              eos_token=special(llama.token_eos()), bos_token=special(llama.token_bos()))
    # The template closes a model turn with its own marker; find it by rendering a finished turn.
    done = fmt(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "\x00"}]).prompt
    end = done.split("\x00", 1)[1].split("\n", 1)[0]

    paragraphs = list(wiki_sentences(DATA.joinpath("wiki.train.raw").read_text(encoding="utf-8").splitlines()))
    cut = len(paragraphs) * 4 // 5
    banned = eval_texts()
    rng = random.Random(0)

    def examples(pool):
        while True:
            sents = rng.choice(pool)
            k = rng.randrange(len(sents))
            clean = " ".join(sents[k:k + rng.choice([1, 1, 2, 3])])
            norm = " ".join(clean.lower().split())
            if any(b in norm for b in banned):
                continue
            text = clean if rng.random() < 0.1 else corrupt(clean, rng)
            if " ".join(text.lower().split()) in banned:
                continue
            step = "polish" if rng.random() < 0.25 else "fix"
            prompt = fmt(messages=[{"role": "system", "content": system_prompt(step, False)},
                                   {"role": "user", "content": f"<text>\n{text}\n</text>"}]).prompt
            yield prompt, clean + end

    tokens, calib = 0, []
    for prompt, answer in examples(paragraphs[:cut]):
        calib.append(prompt + answer + "\n")
        tokens += len(llama.tokenize((prompt + answer).encode(), add_bos=False, special=True))
        if tokens >= args.calib_tokens:
            break
    FIX_CALIB.write_text("".join(calib), encoding="utf-8")
    gen = examples(paragraphs[cut:])
    test = [next(gen) for _ in range(args.test_examples)]
    FIX_TEST.write_text("".join(json.dumps({"prompt": p, "answer": a}) + "\n" for p, a in test), encoding="utf-8")
    print(f"{FIX_CALIB}: {len(calib)} examples, {tokens} tokens; {FIX_TEST}: {len(test)} examples")


def cmd_convert(_):
    need(SRC / "config.json")
    BF16.parent.mkdir(parents=True, exist_ok=True)
    run([PY, LLAMA / "convert_hf_to_gguf.py", SRC, "--outtype", "bf16", "--outfile", BF16])


# The bf16 model (~48 GB) does not fit one GPU: no -ngl here, so llama.cpp's --fit (on by default)
# offloads what fits on the visible GPUs and runs the rest on the CPU.
def cmd_imatrix(args):
    fix = args.corpus == "fix"
    text, out = (FIX_CALIB, IMATRIX_FIX) if fix else (DATA / "wiki.train.raw", IMATRIX)
    need(BF16, text)
    run([BIN / "llama-imatrix", "-m", BF16, "-f", text, "-o", out, "-c", CTX, "--chunks", args.chunks]
        + (["--parse-special"] if fix else []) + (["-t", args.threads] if args.threads else []))


def cmd_kld_base(_):
    need(BF16, DATA / "wiki.test.raw")
    KLD_BASE.parent.mkdir(parents=True, exist_ok=True)
    run([BIN / "llama-perplexity", "-m", BF16, "-f", DATA / "wiki.test.raw", "-c", CTX,
         "--chunks", KLD_CHUNKS, "--kl-divergence-base", KLD_BASE])


def quantize(gate_up, down, out, imatrix=IMATRIX):
    overrides = [f"{GATE_UP}={gate_up}", f"{DOWN}={down}", "token_embd=q6_k"]
    cmd = [BIN / "llama-quantize", "--imatrix", imatrix]
    for o in overrides:
        cmd += ["--tensor-type", o]
    run(cmd + [BF16, out, "Q4_0", str(os.cpu_count())])


def cmd_quantize(_):
    need(BF16, IMATRIX)
    DB.mkdir(parents=True, exist_ok=True)
    wanted = {db_file(t, q) for t, qs in OPTIONS.items() for q in qs}
    for path in sorted(wanted):
        if path.exists():
            print("have", path.name)
            continue
        gu, dn = re.match(r"gu-(\w+)_dn-(\w+)\.gguf", path.name).groups()
        tmp = path.with_suffix(".part")
        quantize(gu, dn, tmp)
        tmp.rename(path)


def cmd_quantize_fix(_):
    need(BF16, IMATRIX_FIX)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "gemma-4-26B-A4B-it-qat-q4_0-imat-fix.gguf"
    tmp = out.with_suffix(".part")
    quantize("q4_0", "q4_0", tmp, IMATRIX_FIX)
    tmp.rename(out)
    print("built", out)


# llama-perplexity can't parse special tokens, so the chat-format KLD runs through llama-cpp-python.
# Only the model turn is scored: the positions that predict the answer tokens and the end-of-turn marker.
def cmd_kld_fix(args):
    import numpy as np
    from llama_cpp import Llama

    model = Path(args.model) if args.model else BF16
    need(model, FIX_TEST)
    if args.model:
        need(KLD_FIX_BASE)
    tests = [json.loads(line) for line in FIX_TEST.read_text(encoding="utf-8").splitlines()]
    llama = Llama(str(model), n_ctx=1024, n_gpu_layers=args.ngl, logits_all=True, verbose=False,
                  n_threads=args.threads, n_threads_batch=args.threads)
    rows = []
    for t in tests:
        n_prompt = len(llama.tokenize(t["prompt"].encode(), add_bos=False, special=True))
        ids = llama.tokenize((t["prompt"] + t["answer"]).encode(), add_bos=False, special=True)
        llama.reset()
        llama.eval(ids)
        logits = np.asarray(llama.scores[n_prompt - 1:len(ids) - 1], dtype=np.float32)
        m = logits.max(axis=1, keepdims=True)
        rows.append(logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))
    logp = np.concatenate(rows)
    if not args.model:
        KLD_FIX_BASE.parent.mkdir(parents=True, exist_ok=True)
        np.save(KLD_FIX_BASE, logp.astype(np.float16))
        print(f"saved {KLD_FIX_BASE}: {len(tests)} examples, {len(logp)} scored tokens")
        return
    ref = np.load(KLD_FIX_BASE).astype(np.float32)
    if ref.shape != logp.shape:
        raise SystemExit(f"{KLD_FIX_BASE} has shape {ref.shape}, this run {logp.shape}: rebuild the base")
    kld = (np.exp(ref) * (ref - logp)).sum(axis=1)
    same = (ref.argmax(axis=1) == logp.argmax(axis=1)).mean()
    r = {"model": str(model), "tokens": len(kld), "kld": float(kld.mean()), "kld_p99": float(np.percentile(kld, 99)),
         "same_top_p": float(100 * same)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"kld-fix-{model.stem}.json").write_text(json.dumps(r, indent=1))
    print(f"{model.name}: fix-set KLD {r['kld']:.6f} (p99 {r['kld_p99']:.4f}), same top {r['same_top_p']:.2f}% "
          f"over {r['tokens']} tokens")


def assemble(choice, out, base_path=None):
    """choice maps tensor name -> db file holding the wanted type of that tensor; the rest comes from the base file."""
    import gguf

    readers = {}

    def reader(path):
        if path not in readers:
            readers[path] = gguf.GGUFReader(path)
        return readers[path]

    base = reader(base_path or db_file(GATE_UP, BASE[GATE_UP]))
    writer = gguf.GGUFWriter(out, arch=base.fields["general.architecture"].contents())
    for field in base.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), field.types[0], sub_type=sub)
    tensors = []
    for t in base.tensors:
        src = choice.get(t.name)
        tensors.append(next(x for x in reader(src).tensors if x.name == t.name) if src else t)
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data)
    writer.close()


def kld(model, gpu, log):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    text = run([BIN / "llama-perplexity", "-m", model, "-f", DATA / "wiki.test.raw", "-c", CTX,
                "--chunks", KLD_CHUNKS, "-ngl", 99, "--kl-divergence-base", KLD_BASE, "--kl-divergence"],
               log=log, env=env)
    mean = re.search(r"Mean\s+KLD:\s+([\d.]+)", text)
    top = re.search(r"Same top p:\s+([\d.]+)", text)
    if not mean:
        raise RuntimeError(f"no KLD in {log}")
    return {"kld": float(mean.group(1)), "same_top_p": float(top.group(1)) if top else None}


def groups(size):
    return [list(range(i, min(i + size, N_LAYERS))) for i in range(0, N_LAYERS, size)]


def tensor_name(tensor, layer):
    return f"blk.{layer}.{tensor}.weight"


def cmd_sweep(args):
    need(KLD_BASE, *(db_file(t, q) for t, qs in OPTIONS.items() for q in qs))
    TMP.mkdir(parents=True, exist_ok=True)
    done = json.loads(SWEEP.read_text()) if SWEEP.exists() else {"group_size": args.group_size, "runs": {}}
    if done["group_size"] != args.group_size:
        raise SystemExit(f"{SWEEP} was made with group size {done['group_size']}; delete it to change")
    jobs = [("base", None, None, None)]
    for gi, layers in enumerate(groups(args.group_size)):
        for tensor, qs in OPTIONS.items():
            for q in qs:
                if q != BASE[tensor]:
                    jobs.append((f"g{gi}-{tensor}-{q}", tensor, q, layers))
    jobs = [j for j in jobs if j[0] not in done["runs"]]
    print(f"{len(jobs)} variants to measure on GPUs {args.gpus}", flush=True)
    free = Queue()
    for g in args.gpus:
        free.put(g)

    def work(job):
        key, tensor, q, layers = job
        gpu = free.get()
        model = TMP / f"{key}.gguf"
        try:
            choice = {tensor_name(tensor, l): db_file(tensor, q) for l in layers} if tensor else {}
            if choice:
                assemble(choice, model)
            else:
                model = db_file(GATE_UP, BASE[GATE_UP])
            r = kld(model, gpu, TMP / f"{key}.log")
            r["bytes"] = model.stat().st_size
            print(f"{key:32} kld {r['kld']:.6f}  top {r['same_top_p']}%  {r['bytes'] / 2**30:.3f} GiB", flush=True)
            return key, r
        finally:
            if tensor:
                model.unlink(missing_ok=True)
            free.put(gpu)

    with ThreadPoolExecutor(len(args.gpus)) as pool:
        for key, r in pool.map(work, jobs):
            done["runs"][key] = r
            SWEEP.write_text(json.dumps(done, indent=1))


def tensor_bytes():
    """Bytes of every searched tensor in every candidate type, read from the database headers."""
    import gguf

    sizes = {}
    for tensor, qs in OPTIONS.items():
        for q in qs:
            for t in gguf.GGUFReader(db_file(tensor, q)).tensors:
                m = re.match(rf"blk\.(\d+)\.{tensor}\.weight", t.name)
                if m:
                    sizes[(tensor, q, int(m.group(1)))] = int(t.n_bytes)
    return sizes


def cmd_allocate(args):
    need(SWEEP)
    sweep = json.loads(SWEEP.read_text())
    runs, size = sweep["runs"], sweep["group_size"]
    base = runs["base"]
    sizes = tensor_bytes()
    budget = int(args.budget_gib * 2**30)
    fixed = base["bytes"] - sum(sizes[(t, BASE[t], l)] for t in OPTIONS for l in range(N_LAYERS))

    # One knapsack item per (group, tensor); each option costs its bytes and adds its measured KLD delta.
    items = []
    for gi, layers in enumerate(groups(size)):
        for tensor, qs in OPTIONS.items():
            opts = []
            for q in qs:
                delta = 0.0 if q == BASE[tensor] else runs[f"g{gi}-{tensor}-{q}"]["kld"] - base["kld"]
                opts.append((q, sum(sizes[(tensor, q, l)] for l in layers), delta))
            items.append((gi, tensor, layers, opts))

    unit = 2**20
    cap = (budget - fixed) // unit
    if cap < 0:
        raise SystemExit(f"budget below the fixed tensors ({fixed / 2**30:.2f} GiB)")
    inf = float("inf")
    best = [0.0] + [inf] * cap
    picks = []
    for _, _, _, opts in items:
        nxt = [inf] * (cap + 1)
        pick = [None] * (cap + 1)
        for c in range(cap + 1):
            if best[c] == inf:
                continue
            for oi, (_, b, d) in enumerate(opts):
                nc = c + -(-b // unit)
                if nc <= cap and best[c] + d < nxt[nc]:
                    nxt[nc], pick[nc] = best[c] + d, (oi, c)
        best = nxt
        picks.append(pick)
    end = min(range(cap + 1), key=lambda c: best[c])
    if best[end] == inf:
        raise SystemExit("no assignment fits the budget")
    chosen, c = [], end
    for (gi, tensor, layers, opts), pick in zip(reversed(items), reversed(picks)):
        oi, c = pick[c]
        chosen.append({"group": gi, "tensor": tensor, "layers": layers, "type": opts[oi][0]})
    chosen.reverse()
    total = fixed + sum(sizes[(x["tensor"], x["type"], l)] for x in chosen for l in x["layers"])
    plan = {"budget_gib": args.budget_gib, "bytes": total, "predicted_kld": base["kld"] + best[end], "assignment": chosen}
    path = WORK / f"plan-{args.budget_gib:g}g.json"
    path.write_text(json.dumps(plan, indent=1))
    for x in chosen:
        print(f"  layers {x['layers'][0]:>2}-{x['layers'][-1]:<2} {x['tensor']:18} {x['type']}")
    print(f"{total / 2**30:.3f} GiB, predicted KLD {plan['predicted_kld']:.6f} -> {path}")


def cmd_build(args):
    path = WORK / f"plan-{args.budget_gib:g}g.json"
    need(path)
    plan = json.loads(path.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"gemma-4-26B-A4B-it-qat-rcolite-{args.budget_gib:g}g.gguf"
    choice = {tensor_name(x["tensor"], l): db_file(x["tensor"], x["type"]) for x in plan["assignment"] for l in x["layers"]}
    assemble(choice, out)
    r = kld(out, args.gpus[0], OUT / f"{out.stem}.kld.log")
    print(f"{out.name}: {out.stat().st_size / 2**30:.3f} GiB, KLD {r['kld']:.6f} "
          f"(predicted {plan['predicted_kld']:.6f}), same top {r['same_top_p']}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    s = sub.add_parser("corpus")
    s.add_argument("--calib-tokens", type=int, default=110_000, help="enough for 200 imatrix chunks of 512")
    s.add_argument("--test-examples", type=int, default=60)
    s.set_defaults(fn=cmd_corpus)
    sub.add_parser("convert").set_defaults(fn=cmd_convert)
    s = sub.add_parser("imatrix")
    s.add_argument("--chunks", type=int, default=200)
    s.add_argument("--corpus", choices=["wiki", "fix"], default="wiki")
    s.add_argument("--threads", type=int, help="CPU threads (default: llama.cpp's)")
    s.set_defaults(fn=cmd_imatrix)
    sub.add_parser("kld-base").set_defaults(fn=cmd_kld_base)
    sub.add_parser("quantize").set_defaults(fn=cmd_quantize)
    sub.add_parser("quantize-fix").set_defaults(fn=cmd_quantize_fix)
    s = sub.add_parser("kld-fix")
    s.add_argument("model", nargs="?", help="GGUF to compare; without it, writes the bf16 reference")
    s.add_argument("--ngl", type=int, default=0, help="layers on the GPU (0 = CPU only)")
    s.add_argument("--threads", type=int, help="CPU threads (default: llama-cpp-python's)")
    s.set_defaults(fn=cmd_kld_fix)
    s = sub.add_parser("sweep")
    s.add_argument("--gpus", type=int, nargs="+", default=[0])
    s.add_argument("--group-size", type=int, default=3)
    s.set_defaults(fn=cmd_sweep)
    s = sub.add_parser("allocate")
    s.add_argument("--budget-gib", type=float, required=True)
    s.set_defaults(fn=cmd_allocate)
    s = sub.add_parser("build")
    s.add_argument("--budget-gib", type=float, required=True)
    s.add_argument("--gpus", type=int, nargs="+", default=[0])
    s.set_defaults(fn=cmd_build)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
