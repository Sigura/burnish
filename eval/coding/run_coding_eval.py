import argparse
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
SYSTEM = (
    "You are an expert software engineer. Reply with the complete contents of the requested file in a single "
    "fenced code block. Do not omit any part of the file."
)
RUNNERS = {
    "python": (["python3", "-s", "-E", "test.py"], "_runner.py", "python3"),
    "bash": (["bash", "test.sh"], "assert.sh", "bash"),
    "kotlin": (["bash", "test.sh"], "TestKit.kt", "kotlinc"),
}
SAMPLING = ("top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")
TEST_TIMEOUT = 60
FEEDBACK_CHARS = 3000


def load_tasks(task_dir: Path, pattern: str, langs: set[str]) -> list[dict]:
    tasks = []
    for path in sorted(task_dir.glob(f"{pattern}/task.toml")):
        task = tomllib.loads(path.read_text())
        task["id"] = path.parent.name
        task["dir"] = path.parent
        if task["lang"] in langs:
            tasks.append(task)
    return tasks


def extract_code(reply: str) -> str | None:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    blocks = re.findall(r"```[^\n]*\n(.*?)```", reply, flags=re.S)
    return blocks[-1] if blocks else None


def run_test_cmd(cmd: list[str], work: Path) -> subprocess.CompletedProcess:
    """subprocess.run with the tests in their own process group, killed as a whole afterwards: a timeout alone kills
    only the test runner, and a solution it started kept running (one awk grew to 82 GB)."""
    proc = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run_tests(task: dict, code: str) -> tuple[bool, str]:
    cmd, helper, _ = RUNNERS[task["lang"]]
    with tempfile.TemporaryDirectory(prefix=f"ce-{task['id']}-") as tmp:
        work = Path(tmp)
        shutil.copy(ROOT / "lib" / helper, work / helper)
        for f in task["dir"].iterdir():
            if f.name.lower().startswith(("test", "fixture")):
                shutil.copy(f, work / f.name)
        (work / task["file"]).write_text(code)
        try:
            proc = run_test_cmd(cmd, work)
        except subprocess.TimeoutExpired:
            return False, f"tests timed out after {TEST_TIMEOUT} s"
        output = (proc.stdout + proc.stderr).strip()
        return proc.returncode == 0, output


def chat(url: str, messages: list[dict], args) -> dict:
    body = {"messages": messages, "temperature": args.temperature, "seed": args.seed, "max_tokens": args.max_tokens}
    for key in SAMPLING:
        if getattr(args, key) is not None:
            body[key] = getattr(args, key)
    if args.model:
        body["model"] = args.model
    if args.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("CODING_EVAL_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['CODING_EVAL_API_KEY']}"
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(), headers)
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.load(resp)
    data["_seconds"] = time.monotonic() - start
    return data


REVIEW_SYSTEM = (
    "You are a strict code reviewer. You get a task description and code written by another model.\n"
    "Find real defects only: wrong behavior versus the task, crashes, unhandled edge cases the task implies, security\n"
    "problems, broken commands. Ignore style, naming and missing comments.\n"
    "Answer in at most 10 lines and nothing else: either the single word OK, or one line per defect in the form\n"
    "line: problem -> fix"
)


def review(task_prompt: str, code: str, args) -> tuple[str, float]:
    """Independent review by another model: sees only the task and the code, answers OK or a few defect lines."""
    numbered = "\n".join(f"{i:4} {line}" for i, line in enumerate(code.splitlines(), 1))
    body = {"messages": [{"role": "system", "content": REVIEW_SYSTEM},
                         {"role": "user", "content": f"Task: {task_prompt}\n\n{numbered}"}],
            "max_tokens": 600, "temperature": 0.2, "seed": args.seed,
            "chat_template_kwargs": {"enable_thinking": False}}
    if args.review_model:
        body["model"] = args.review_model
    headers = {"Content-Type": "application/json"}
    if os.environ.get("CODING_EVAL_REVIEW_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['CODING_EVAL_REVIEW_KEY']}"
    req = urllib.request.Request(f"{args.review_url}/v1/chat/completions", json.dumps(body).encode(), headers)
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        data = json.load(resp)
    return (data["choices"][0]["message"].get("content") or "").strip(), time.monotonic() - start


def solve(task: dict, args) -> dict:
    prompt = f"{task['prompt'].strip()}\n\nWrite the file `{task['file']}`."
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    attempts = []
    for attempt in range(args.attempts):
        try:
            data = chat(args.url, messages, args)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            # a runaway earlier reply can push the retry prompt past the server context: a failed attempt, not a crash
            if e.code != 400 or "context" not in body:
                raise
            attempts.append({"passed": False, "capped": True, "output": f"server rejected the request: {body[:300]}",
                             "seconds": 0.0, "completion_tokens": 0, "reply": ""})
            break
        choice = data["choices"][0]
        reply = choice["message"].get("content") or ""
        capped = choice.get("finish_reason") == "length"
        code = extract_code(reply)
        reviewed = None
        if code is not None and args.review_url:
            verdict, review_seconds = review(task["prompt"].strip(), code, args)
            reviewed = {"verdict": verdict, "seconds": round(review_seconds, 1)}
            data["_seconds"] += review_seconds
            if verdict.rstrip(".").upper() != "OK":
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"A reviewer reported:\n```\n{verdict}\n```\nFix the real defects "
                                                         f"(ignore findings that are wrong) and reply with the corrected "
                                                         f"complete `{task['file']}`."}]
                revised = chat(args.url, messages, args)
                data["_seconds"] += revised["_seconds"]
                reply = revised["choices"][0]["message"].get("content") or ""
                capped = revised["choices"][0].get("finish_reason") == "length"
                code = extract_code(reply)
        if code is None:
            passed = False
            output = f"reply cut off at --max-tokens {args.max_tokens}" if capped else "no fenced code block in the reply"
        else:
            passed, output = run_tests(task, code)
        usage = data.get("usage", {})
        attempts.append({"passed": passed, "capped": capped, "output": output[-FEEDBACK_CHARS:],
                         "seconds": round(data["_seconds"], 1),
                         "completion_tokens": usage.get("completion_tokens"), "reply": reply,
                         **({"review": reviewed} if reviewed else {})})
        if passed:
            break
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": f"The tests failed:\n```\n{output[-FEEDBACK_CHARS:]}\n```\n"
                                                 f"Reply with the corrected complete `{task['file']}`."}]
    return {"id": task["id"], "lang": task["lang"], "attempts": attempts}


def test_fraction(attempt: dict) -> float:
    if attempt["passed"]:
        return 1.0
    m = re.findall(r"^(\d+)/(\d+) passed$", attempt["output"], flags=re.M)
    return int(m[-1][0]) / int(m[-1][1]) if m and int(m[-1][1]) else 0.0


def summarize(results: list[dict], attempts: int) -> dict:
    by_lang: dict[str, dict] = {}
    for r in results:
        for key in ("all", r["lang"]):
            s = by_lang.setdefault(key, {"tasks": 0, **{f"pass@{k}": 0 for k in range(1, attempts + 1)},
                                         **{f"tests@{k}": 0.0 for k in range(1, attempts + 1)},
                                         "capped": 0, "seconds": 0.0, "completion_tokens": 0})
            s["tasks"] += 1
            for k in range(1, attempts + 1):
                s[f"pass@{k}"] += any(a["passed"] for a in r["attempts"][:k])
                s[f"tests@{k}"] = round(s[f"tests@{k}"] + max(test_fraction(a) for a in r["attempts"][:k]), 3)
            s["capped"] += sum(a.get("capped", False) for a in r["attempts"])
            s["seconds"] = round(s["seconds"] + sum(a["seconds"] for a in r["attempts"]), 1)
            s["completion_tokens"] += sum(a["completion_tokens"] or 0 for a in r["attempts"])
    return by_lang


def main():
    parser = argparse.ArgumentParser(description="Coding eval against an OpenAI-compatible llama-server endpoint.")
    parser.add_argument("--url", help="e.g. http://127.0.0.1:8082")
    parser.add_argument("--model", help="model name for router-mode servers")
    parser.add_argument("--label", default="run", help="result file suffix")
    parser.add_argument("--tasks", default="*", help="glob over task ids")
    parser.add_argument("--exclude", help="glob over task ids to skip, e.g. '*_e_*'")
    parser.add_argument("--task-dir", default=str(ROOT / "tasks"), help="directory holding the task folders")
    parser.add_argument("--langs", default="python,bash")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    for key in SAMPLING:
        parser.add_argument(f"--{key.replace('_', '-')}", type=int if key == "top_k" else float,
                            help="sent to the server only when given")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--review-url", help="reviewer endpoint: each attempt's code is reviewed and revised once "
                                             "before the tests run (key: env CODING_EVAL_REVIEW_KEY)")
    parser.add_argument("--review-model", help="reviewer model name for router-mode servers")
    parser.add_argument("--no-think", action="store_true", help="send chat_template_kwargs.enable_thinking=false")
    parser.add_argument("--ref", action="store_true", help="check reference solutions instead of querying a model")
    parser.add_argument("--rescore", metavar="RESULT_JSON",
                        help="re-run the current tests on the code stored in a result file (no model calls)")
    args = parser.parse_args()

    langs = set(args.langs.split(","))
    for lang in langs:
        tool = RUNNERS[lang][2]
        if shutil.which(tool) is None:
            parser.error(f"--langs {lang} needs `{tool}` on PATH")
    tasks = load_tasks(Path(args.task_dir), args.tasks, langs)
    if args.exclude:
        tasks = [t for t in tasks if not fnmatch.fnmatch(t["id"], args.exclude)]
    if not tasks:
        parser.error("no tasks matched")

    if args.ref:
        bad = 0
        for task in tasks:
            ref = next(task["dir"].glob("ref.*"))
            passed, output = run_tests(task, ref.read_text())
            bad += not passed
            print(f"{'ok  ' if passed else 'FAIL'} {task['id']}" + ("" if passed else f"\n{output}"))
        print(f"{len(tasks) - bad}/{len(tasks)} reference solutions pass")
        raise SystemExit(1 if bad else 0)

    if args.rescore:
        src = Path(args.rescore)
        data = json.loads(src.read_text())
        by_id = {t["id"]: t for t in tasks}
        results = [r for r in data["results"] if r["id"] in by_id]
        for r in results:
            for a in r["attempts"]:
                code = extract_code(a["reply"])
                if code is not None:
                    a["passed"], output = run_tests(by_id[r["id"]], code)
                    a["output"] = output[-FEEDBACK_CHARS:]
            print(f"{r['id']:<28} {''.join('+' if a['passed'] else '-' for a in r['attempts'])}", flush=True)
        attempts = data["args"]["attempts"]
        data.update(summary=summarize(results, attempts), results=results, rescored_from=src.name)
        out = src.with_name(f"{src.stem}-rescored.json")
        out.write_text(json.dumps(data, indent=1))
        print_summary(data["summary"], out)
        return

    if not args.url:
        parser.error("--url is required unless --ref or --rescore")
    results = []
    for task in tasks:
        r = solve(task, args)
        results.append(r)
        marks = "".join("+" if a["passed"] else "-" for a in r["attempts"])
        print(f"{task['id']:<28} {marks:<3} {sum(a['seconds'] for a in r['attempts']):7.1f} s", flush=True)
    summary = summarize(results, args.attempts)
    out = ROOT / "results" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "results": results}, indent=1))
    print_summary(summary, out)


def print_summary(summary: dict, out: Path):
    for lang, s in summary.items():
        passes = "  ".join(f"{k} {v}/{s['tasks']}" for k, v in s.items() if k.startswith(("pass@", "tests@")))
        print(f"{lang:<8} {passes}  {s['seconds']} s  {s['completion_tokens']} tokens  {s['capped']} attempts capped")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
