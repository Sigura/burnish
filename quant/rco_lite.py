"""RCO-lite: budget-constrained per-tensor quant allocation for gemma4:26b-a4b with stock llama.cpp tools.

Pipeline (each step is a subcommand, run in order):
  data      wikitext-2 train (imatrix calibration) and test (KL divergence)
  convert   bf16 QAT checkpoint -> bf16 GGUF
  imatrix   importance matrix from the bf16 GGUF
  kld-base  reference log-probs of the bf16 model
  quantize  one full GGUF per candidate expert type (the tensor database)
  sweep     KL divergence of the base mix with one layer group swapped to each alternative
  allocate  multiple-choice knapsack over the sweep deltas under a size budget
  build     assemble the allocated mix and measure its real KL divergence

Only the MoE expert tensors are searched; they are 90% of the weights. Everything else keeps the
layout of Google's QAT Q4_0 release (Q4_0, token embeddings Q6_K).
"""

import argparse
import json
import os
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
DATA = WORK / "data"
KLD_BASE = WORK / "kld/bf16.kld"
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


def cmd_convert(_):
    need(SRC / "config.json")
    BF16.parent.mkdir(parents=True, exist_ok=True)
    run([PY, LLAMA / "convert_hf_to_gguf.py", SRC, "--outtype", "bf16", "--outfile", BF16])


# The bf16 model (~48 GB) does not fit one GPU: no -ngl here, so llama.cpp's --fit (on by default)
# offloads what fits on the visible GPUs and runs the rest on the CPU.
def cmd_imatrix(args):
    need(BF16, DATA / "wiki.train.raw")
    run([BIN / "llama-imatrix", "-m", BF16, "-f", DATA / "wiki.train.raw", "-o", IMATRIX,
         "-c", CTX, "--chunks", args.chunks])


def cmd_kld_base(_):
    need(BF16, DATA / "wiki.test.raw")
    KLD_BASE.parent.mkdir(parents=True, exist_ok=True)
    run([BIN / "llama-perplexity", "-m", BF16, "-f", DATA / "wiki.test.raw", "-c", CTX,
         "--chunks", KLD_CHUNKS, "--kl-divergence-base", KLD_BASE])


def quantize(gate_up, down, out):
    overrides = [f"{GATE_UP}={gate_up}", f"{DOWN}={down}", "token_embd=q6_k"]
    cmd = [BIN / "llama-quantize", "--imatrix", IMATRIX]
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
    sub.add_parser("convert").set_defaults(fn=cmd_convert)
    s = sub.add_parser("imatrix")
    s.add_argument("--chunks", type=int, default=200)
    s.set_defaults(fn=cmd_imatrix)
    sub.add_parser("kld-base").set_defaults(fn=cmd_kld_base)
    sub.add_parser("quantize").set_defaults(fn=cmd_quantize)
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
