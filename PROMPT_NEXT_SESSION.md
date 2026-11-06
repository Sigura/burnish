Project: /home/sigura/work/burnish (burnish, an offline text refiner using in-process llama-cpp-python and LanguageTool).

Status: make test: 118 passed, 1 skipped. Latest eval: eval/results/20261106-131039.json (fix mode, hybrid exact / over-correction): gemma3:12b 95%/1%, qwen3:14b 92%/0%, gemma3:4b gpu 83%/3%, gemma3:4b cpu 86%/2%, qwen3:8b 82%/0%, phi4-mini 76%/0%. Defaults unchanged: gemma3:12b (>=12 GB), gemma3:4b otherwise and on CPU. hybrid >= llm-only for every model; no curly quotes introduced; qwen3:8b uses 5652 MiB, under its 6144 MiB tier.

Prompt (src/burnish/pipeline.py): the [[Pn]] rule is included only when the text has protected spans; abbreviations, acronyms, slang and technical terms are kept as written; the fix prompt says to pick the correction that changes the fewest words. keep_item_endings() drops a period the LLM added to a list/heading/quote line.

Running evals in the devbox container: ~/.local/share/burnish is mounted read-only at its host path. Java (devbox install default-jre-headless) and the LanguageTool server are lost on every container restart; start LanguageTool with
java -Xmx2g -cp ~/.local/share/burnish/languagetool/current/languagetool-server.jar org.languagetool.server.HTTPServer --config ~/.local/share/burnish/languagetool/current/server.properties --port 8081
devbox job containers can't reach it, so run make eval in Claude's container or on the host.

Open issues:
- gemma3:12b fix mode: "I am agree" -> "I am in agreement" (all mode gets "I agree"). Accepted for now.
- gemma3:4b: "He explained me the problem." -> "He explained it to me the problem."; drops inline [[Pn]] in polish mode (accepted).
- phi4-mini misses "Its" -> "It's"; 3 guard rejections in fix mode.
- The CPU tier was measured on a CUDA build, which still offloads to the GPU (about 1.3 GB VRAM used), so CPU timings are optimistic.
- X11 hotkeys (xdotool/ydotool) not tested on a real X11 desktop; the host lacked xdotool, xclip and wl-clipboard.
- Windows/macOS desktop integration not implemented (src/burnish/desktop.py).
