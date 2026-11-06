# burnish

Offline refinement of English text (grammar, spelling, fluency) from any app. It combines a local
LLM (llama.cpp, run inside the backend) with a self-hosted LanguageTool server. Nothing leaves the machine after setup.

## Setup

```sh
make install     # asks before every sudo and before large downloads
make test        # unit tests; the integration test is skipped when services are down
make eval        # compares models and the pipeline on eval/sentences.tsv
make uninstall
```

Requirements: systemd, `curl`, and a package manager among apt, dnf, pacman and zypper (without one,
the installer lists what to install by hand). `uv`, a C++ compiler, the CUDA toolkit, Java and the desktop tools are installed
on request.

## Modes

| Mode | What it does |
|---|---|
| `all` (default) | `fix`, then `polish` on the result. Two LLM passes, so the slowest. |
| `fix` | Minimal grammar and spelling edits. |
| `polish` | Improves fluency and clarity, keeps the meaning. |

The default applies to the API, the CLI, the Vim mappings and the Chrome extension. The two hotkeys
are bound to `fix` and `polish` explicitly.

## How a request flows

1. Code blocks, inline code, URLs, paths, emails and markdown line markers become `[[P0]]` tokens.
2. The LLM rewrites the masked text at temperature 0.
3. The rewrite is discarded if it was truncated, lost a token, changed `**`/`__`/`~~` counts, or
   drifted past the `[drift]` char-ratio threshold.
4. LanguageTool checks the result and applies `apply_categories` fixes. Spelling fixes skip mixed-case
   words, identifiers and capitalized words in mid-sentence. Add your own to `ignore_words`.

When a rewrite is discarded, the response has `model: "languagetool"` and an issue saying why. If
a model can't run or LanguageTool is down, the request fails with 503 instead of returning a partial result.

```sh
curl -s 127.0.0.1:8765/refine -H 'Content-Type: application/json' \
  -d '{"text": "She go home.", "mode": "fix"}'
curl -s 127.0.0.1:8765/health
```

## Configuration

`~/.config/burnish/config.toml` is merged over `src/burnish/default_config.toml`. Restart after editing:
`systemctl --user restart burnish`.

The model is chosen from free VRAM reported by `nvidia-smi`, cached for 30 s. VRAM held by the
backend's own loaded model counts as free, otherwise a loaded model would demote the next request to a lower
tier. Without an NVIDIA GPU, the CPU tier is used.

The default models come from `make eval` on an RTX 4090 Laptop GPU (16 GB). Run it on your machine
to check them: it prints a suggested model per tier, using the fix-mode quality scores and the `all`-mode p95 latency against the target
(3 s on GPU, 8 s on CPU). Sentences take two LLM passes in `all` mode, and the first request after
`keep_alive` expires also pays for loading the model.

## Models

The backend loads GGUF weights in-process through `llama-cpp-python`. No model server is involved, and
downloads only happen when you ask for them.

```sh
burnish-models list
burnish-models pull qwen3:8b     # resumes if interrupted, verifies the SHA-256
burnish-models pull --all        # every model the config uses
```

Weights go to `~/.local/share/burnish/models/` (`models_dir` changes it). Each `[models."name"]` entry
is a Hugging Face repo and a file pattern. If a pattern doesn't match exactly one file, the error lists
what's available. Set `HF_TOKEN` for gated repos.

A loaded model is freed after `keep_alive_seconds` without requests (60 s on GPU, so VRAM goes back to
your ML jobs). llama.cpp is compiled with CUDA during `make install` when `nvidia-smi` and `nvcc` are
present; `/health` reports the result as `gpu_offload`. To rebuild by hand:

```sh
CMAKE_ARGS="-DGGML_CUDA=on" uv sync --reinstall-package llama-cpp-python --refresh-package llama-cpp-python
```

## CLI

```sh
echo 'She go home.' | burnish
burnish --mode fix --diff < draft.md
burnish --model phi4-mini < draft.md
EDITOR=burnish            # edits the given file in place
```

On failure, `burnish` writes its input back to stdout and exits 1, so a Vim filter never loses text.
The error goes to stderr, or to a notification and `~/.local/state/burnish/cli.log` when stderr is
merged into stdout, as it is inside Vim.

## Vim and Neovim

Add `vim/burnish.vim` to your config, or copy its two mappings. Vim filters whole lines, so a partial
line selection still refines the full line.

## Global hotkeys

| Shortcut | Action |
|---|---|
| `Ctrl+Alt+P` | fix |
| `Ctrl+Alt+Shift+P` | polish |

The installer creates `~/.local/share/applications/burnish-{fix,polish}.desktop` and registers them in
KDE's `kglobalshortcutsrc`. You can see and change them in System Settings > Shortcuts. On other
desktops, the installer prints the commands to bind yourself.

| | X11 | Wayland |
|---|---|---|
| Read selection | `Ctrl+C` via `xdotool`, read with `xclip` | PRIMARY via `wl-paste --primary` |
| Paste | `Ctrl+V` via `xdotool` | `Ctrl+V` via `ydotool`, or `wtype` where supported |
| Terminal detection | window class via `xdotool` | only with `kdotool` installed |

The session type comes from `$XDG_SESSION_TYPE`.
On X11, set `x11_keystrokes = "ydotool"` under `[hotkey]` if `Ctrl+C`/`Ctrl+V` go missing with a
non-Latin keyboard layout: `xdotool` maps keys through the active layout, `ydotool` sends physical keys.
It needs the `burnish-ydotoold` service, which `make install` sets up when you answer yes to ydotool.

- **Clipboard.** Saved and restored as one type: PNG if present, otherwise text. Other formats, such
  as HTML, are lost. Klipper's history will show the temporary marker and the result.
- **Terminals.** `Ctrl+C` would interrupt the running program and a paste could execute a command, so
  terminals never receive keystrokes. The selection is read from PRIMARY and the result goes to the
  clipboard, with a notification. Add other terminals to `terminal_classes`.
- **Focus changes.** If you switch windows while a request runs, the result goes to the clipboard
  instead of being pasted.
- **Wayland without `kdotool`.** Terminals can't be detected, so the paste goes out anyway (`Ctrl+V`
  does nothing useful in Konsole). The result is saved to `~/.local/state/burnish/last_result.txt`.
- **Wayland with nothing selected.** PRIMARY keeps the last selection, so an old selection can be
  refined and pasted.

## Chrome extension

1. Open `chrome://extensions` and enable Developer mode.
2. Choose Load unpacked and select the `extension/` directory.

Use the right-click item "Burnish selection", or `Alt+Shift+P` (change it at
`chrome://extensions/shortcuts`). Chrome doesn't allow `Ctrl+Alt` shortcuts for extensions. Selections
in read-only page text are copied to the clipboard instead of replaced. The `<all_urls>` permission is
needed to reach editors inside cross-origin iframes such as Google Chat; the extension only runs when
you trigger it. The mode is the `MODE` constant in `extension/background.js`.

## Troubleshooting

| Symptom | Check |
|---|---|
| Backend not reachable | `systemctl --user status burnish`, `journalctl --user -u burnish` |
| 503 from `/refine` | The message names the cause; `curl -s 127.0.0.1:8765/health` shows service state |
| Model not downloaded | `burnish-models pull <name>` |
| Slow, `gpu_offload` is false | llama.cpp was built without CUDA; run the rebuild command above |
| `make eval` runs out of VRAM | Stop the backend first: `systemctl --user stop burnish` |
| LanguageTool down | `journalctl --user -u burnish-languagetool`; it can take a while to start with n-grams |
| Shortcut does nothing | System Settings > Shortcuts, then run `burnish-hotkey --mode fix` in a terminal |
| Ctrl+Alt+C reaches the app | Raise `settle_ms`; the shortcut keys were still held |
| Old clipboard gets pasted | Raise `paste_delay_ms` |
| "No text selected" with a selection | Raise `copy_timeout_ms` |
| `ydotool` fails on Wayland | `systemctl --user status burnish-ydotoold`, access to `/dev/uinput`; 1.x syntax is required |
| `wtype` fails on KDE | KWin doesn't support its protocol; use `ydotool` |
| Rewrites often discarded | Check `issues` in the response and adjust `[drift]` |
| Names get "corrected" | Add them to `ignore_words` |
| Model keeps switching | Compare `free_vram_mb` in `/health` with the tier thresholds |

Logs are in `~/.local/state/burnish/`. They record sizes, models and timings, not your text.

## Portability

The backend, CLI, pipeline, eval and extension have no Linux-specific code, and llama.cpp and
LanguageTool's Java server are cross-platform. Linux-only parts:

- `src/burnish/desktop_linux.py`: clipboard, keystrokes, notifications and the instance lock. A port
  adds a sibling module and a branch in `desktop.py`. Windows isn't implemented yet.
- `install.sh`, `uninstall.sh` and `systemd/`: a port needs its own installer and service registration.
- The config and state paths at the top of `src/burnish/config.py`.
