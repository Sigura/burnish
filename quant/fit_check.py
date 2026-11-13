"""Loads a GGUF with burnish's GPU-tier options and reports the CUDA buffers llama.cpp allocates.

Run with burnish's venv: .venv-devbox/bin/python quant/fit_check.py MODEL.gguf [--n-ctx 8192]
The sum is what the model needs on top of the CUDA context (~300-500 MiB, not counted here).
"""

import argparse
import os
import re
import subprocess
import sys

LOAD = """
import sys
from llama_cpp import Llama
m = Llama(sys.argv[1], n_ctx=int(sys.argv[2]), n_gpu_layers=-1, flash_attn=True, type_k=8, type_v=8, verbose=True)
m.create_completion("Fix this sentence: she go to school.", max_tokens=16, temperature=0)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--n-ctx", type=int, default=8192)
    args = p.parse_args()
    proc = subprocess.run([sys.executable, "-c", LOAD, args.model, str(args.n_ctx)],
                          capture_output=True, text=True, env=os.environ)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(log[-3000:])
        raise SystemExit(f"load failed with exit code {proc.returncode}")
    total = 0.0
    for line in log.splitlines():
        m = re.search(r"(CUDA\d+)\s+(model buffer|KV buffer|compute buffer|RS buffer)[^=]*=\s*([\d.]+) MiB", line)
        if m:
            total += float(m.group(3))
            print(f"  {m.group(1)} {m.group(2):15} {float(m.group(3)):9.1f} MiB")
    print(f"total CUDA buffers: {total:.0f} MiB (n_ctx {args.n_ctx}); add ~300-500 MiB of CUDA context")


main()
