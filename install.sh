#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$HOME/.local/share/burnish"
LT_ROOT="$DATA_DIR/languagetool"
LT_DIR="$LT_ROOT/current"
NGRAM_DIR="$DATA_DIR/ngrams"
NGRAM_URL="https://languagetool.org/download/ngram-data/ngrams-en-20150817.zip"
NGRAM_MIN_FREE_GB=30
UNIT_DIR="$HOME/.config/systemd/user"
APP_DIR="$HOME/.local/share/applications"
BIN_DIR="$HOME/.local/bin"
CONFIG_FILE="$HOME/.config/burnish/config.toml"

step() { printf '\n==> %s\n' "$*"; }
ask() {
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty
    [[ "$reply" =~ ^[Yy]$ ]]
}
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }

PKG_MANAGER=""
for candidate in apt-get dnf pacman zypper; do
    if command -v "$candidate" >/dev/null; then
        PKG_MANAGER="$candidate"
        break
    fi
done

package_for() {
    case "$PKG_MANAGER:$1" in
        apt-get:java) echo default-jre-headless ;;
        dnf:java) echo java-latest-openjdk-headless ;;
        pacman:java) echo jre-openjdk-headless ;;
        zypper:java) echo java-21-openjdk-headless ;;
        apt-get:notify-send) echo libnotify-bin ;;
        zypper:notify-send) echo libnotify-tools ;;
        *:notify-send) echo libnotify ;;
        *:wl-copy) echo wl-clipboard ;;
        apt-get:nvcc) echo nvidia-cuda-toolkit ;;
        pacman:nvcc) echo cuda ;;
        pacman:g++) echo gcc ;;
        dnf:g++ | zypper:g++) echo gcc-c++ ;;
        *) echo "$1" ;;
    esac
}

# Takes command names, so the check is the same on every distribution.
ensure_commands() {
    local packages=() missing=()
    for tool in "$@"; do
        if ! command -v "$tool" >/dev/null; then
            missing+=("$tool")
            packages+=("$(package_for "$tool")")
        fi
    done
    [[ ${#missing[@]} -eq 0 ]] && return 0

    local install
    case "$PKG_MANAGER" in
        apt-get) install=(sudo apt-get install -y) ;;
        dnf) install=(sudo dnf install -y) ;;
        pacman) install=(sudo pacman -S --needed --noconfirm) ;;
        zypper) install=(sudo zypper install -y) ;;
        *)
            echo "No supported package manager found. Please install: ${missing[*]}"
            return 0
            ;;
    esac
    if ask "Run '${install[*]} ${packages[*]}'?"; then
        "${install[@]}" "${packages[@]}"
    else
        echo "Skipped: ${packages[*]}. Install them yourself for this feature to work."
    fi
}

step "Checking prerequisites"
ensure_commands curl unzip
for tool in curl unzip systemctl; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
if ! command -v uv >/dev/null; then
    ask "uv is missing. Install it from https://astral.sh/uv/install.sh (no sudo)?" \
        || { echo "uv is required." >&2; exit 1; }
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$BIN_DIR:$PATH"
fi

step "Building the Python environment and llama.cpp"
ensure_commands cmake g++
BUILD_GPU=no
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    case "$PKG_MANAGER" in
        apt-get | pacman) ensure_commands nvcc ;;
    esac
    if command -v nvcc >/dev/null; then
        BUILD_GPU=yes
    else
        echo "WARNING: the CUDA toolkit (nvcc) is missing, so llama.cpp will run on CPU only."
    fi
fi
gpu_build_active() {
    (cd "$REPO" && uv run --no-sync python -c \
        "from llama_cpp import llama_supports_gpu_offload as f; raise SystemExit(0 if f() else 1)") >/dev/null 2>&1
}
cd "$REPO"
if [[ "$BUILD_GPU" == yes ]] && ! gpu_build_active; then
    echo "Compiling llama.cpp with CUDA. This takes several minutes."
    CMAKE_ARGS="-DGGML_CUDA=on" uv sync --reinstall-package llama-cpp-python --refresh-package llama-cpp-python
    gpu_build_active || echo "WARNING: the CUDA build has no GPU support. Models will run on CPU."
else
    uv sync
fi

step "Installing the default config"
mkdir -p "$(dirname "$CONFIG_FILE")"
if [[ -e "$CONFIG_FILE" ]]; then
    echo "Keeping existing $CONFIG_FILE"
else
    cp "$REPO/src/burnish/default_config.toml" "$CONFIG_FILE"
    echo "Wrote $CONFIG_FILE"
fi

step "Linking commands into $BIN_DIR"
mkdir -p "$BIN_DIR"
for command in burnish burnish-hotkey burnish-models; do
    ln -sf "$REPO/.venv/bin/$command" "$BIN_DIR/$command"
done
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) echo "NOTE: add $BIN_DIR to PATH in ~/.zshenv so Vim's filter can find burnish." ;;
esac

step "Downloading model weights"
if ask "Download all candidate models from Hugging Face (about 30 GB)? 'make eval' compares them."; then
    "$REPO/.venv/bin/burnish-models" pull --all
else
    echo "Skipped. Download them later with: burnish-models pull --all"
fi

step "Setting up LanguageTool"
ensure_commands java
command -v java >/dev/null || { echo "Java is required for LanguageTool." >&2; exit 1; }
mkdir -p "$LT_ROOT"
if [[ ! -e "$LT_DIR/languagetool-server.jar" ]]; then
    LT_ZIP="$(mktemp --suffix=.zip)"
    curl -fL -o "$LT_ZIP" https://languagetool.org/download/LanguageTool-stable.zip
    unzip -q -o "$LT_ZIP" -d "$LT_ROOT"
    rm -f "$LT_ZIP"
    LT_EXTRACTED="$(find "$LT_ROOT" -maxdepth 1 -type d -name 'LanguageTool-*' | sort -V | tail -1)"
    [[ -n "$LT_EXTRACTED" ]] || { echo "LanguageTool archive had an unexpected layout." >&2; exit 1; }
    ln -sfn "$LT_EXTRACTED" "$LT_DIR"
fi

AVAILABLE_GB="$(free_gb "$DATA_DIR")"
if [[ ! -d "$NGRAM_DIR/en" ]]; then
    if (( AVAILABLE_GB < NGRAM_MIN_FREE_GB )); then
        echo "Skipping n-grams: ${AVAILABLE_GB} GB free, need ${NGRAM_MIN_FREE_GB} GB."
    elif ask "Download English n-grams (about 8 GB, more once unpacked; ${AVAILABLE_GB} GB free)?"; then
        mkdir -p "$NGRAM_DIR"
        NGRAM_ZIP="$NGRAM_DIR/ngrams-en.zip"
        curl -fL -C - -o "$NGRAM_ZIP" "$NGRAM_URL"
        unzip -q -o "$NGRAM_ZIP" -d "$NGRAM_DIR"
        rm -f "$NGRAM_ZIP"
    fi
fi
{
    echo "cacheSize=1000"
    if [[ -d "$NGRAM_DIR/en" ]]; then echo "languageModel=$NGRAM_DIR"; fi
} >"$LT_DIR/server.properties"

step "Installing hotkey tools"
ensure_commands xdotool xclip notify-send wl-copy

WAYLAND_PASTE=no
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]] \
    || ask "Set up ydotool too (Wayland pasting, or x11_keystrokes = \"ydotool\" on X11)? It needs write access to /dev/uinput."; then
    ensure_commands ydotool
    if command -v ydotool >/dev/null; then
        WAYLAND_PASTE=yes
        if [[ ! -w /dev/uinput ]]; then
            cat <<'NOTE'
ydotool needs /dev/uinput. Granting it through the 'input' group lets any program you run
inject keystrokes and read input devices.
NOTE
            if ask "Run sudo to add a udev rule for uinput and add $USER to the 'input' group?"; then
                echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' \
                    | sudo tee /etc/udev/rules.d/60-burnish-uinput.rules >/dev/null
                sudo udevadm control --reload-rules
                sudo udevadm trigger --name-match=uinput
                sudo usermod -aG input "$USER"
                echo "Log out and back in for the group change to apply."
            fi
        fi
    fi
fi

step "Registering systemd user units"
mkdir -p "$UNIT_DIR"
render_unit() {
    sed -e "s|@REPO@|$REPO|g" -e "s|@LT_DIR@|$LT_DIR|g" "$REPO/systemd/$1.service.in" >"$UNIT_DIR/$1.service"
}
render_unit burnish
render_unit burnish-languagetool
UNITS=(burnish-languagetool burnish)
if [[ "$WAYLAND_PASTE" == yes ]]; then
    render_unit burnish-ydotoold
    UNITS+=(burnish-ydotoold)
fi
systemctl --user daemon-reload
for unit in "${UNITS[@]}"; do
    systemctl --user enable "$unit.service"
    systemctl --user restart "$unit.service"
done

step "Registering global shortcuts"
mkdir -p "$APP_DIR"
write_desktop() {
    local id="$1" label="$2" mode="$3" keys="$4"
    cat >"$APP_DIR/$id.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$label
Exec=$REPO/.venv/bin/burnish-hotkey --mode $mode
NoDisplay=true
Terminal=false
StartupNotify=false
X-KDE-Shortcuts=$keys
DESKTOP
    # Plasma 5 and 6 store these differently.
    if command -v kwriteconfig6 >/dev/null; then
        kwriteconfig6 --file kglobalshortcutsrc --group services --group "$id.desktop" --key _launch "$keys"
    elif command -v kwriteconfig5 >/dev/null; then
        kwriteconfig5 --file kglobalshortcutsrc --group "$id.desktop" --key _k_friendly_name "$label"
        kwriteconfig5 --file kglobalshortcutsrc --group "$id.desktop" --key _launch "$keys,none,$label"
    fi
}
write_desktop burnish-fix "Burnish: fix" fix "Ctrl+Alt+P"
write_desktop burnish-polish "Burnish: polish" polish "Ctrl+Alt+Shift+P"
if [[ "${XDG_CURRENT_DESKTOP:-}" == *KDE* ]]; then
    if systemctl --user is-active --quiet plasma-kglobalaccel.service; then
        systemctl --user restart plasma-kglobalaccel.service
    else
        echo "Log out and back in so KDE picks up the shortcuts."
    fi
else
    cat <<NOTE
This isn't KDE, so bind these commands in your desktop's keyboard settings:
  Ctrl+Alt+P        $BIN_DIR/burnish-hotkey --mode fix
  Ctrl+Alt+Shift+P  $BIN_DIR/burnish-hotkey --mode polish
NOTE
fi

step "Waiting for the backend"
for _ in $(seq 1 60); do
    curl -fs http://127.0.0.1:8765/health >/dev/null && break
    sleep 1
done
curl -fs http://127.0.0.1:8765/health || { echo "Backend did not start. Check: journalctl --user -u burnish" >&2; exit 1; }
echo

step "Done. Try: echo 'She go home.' | burnish"
