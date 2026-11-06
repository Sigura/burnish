#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
APP_DIR="$HOME/.local/share/applications"

step() { printf '\n==> %s\n' "$*"; }
ask() {
    local reply
    read -r -p "$1 [y/N] " reply </dev/tty
    [[ "$reply" =~ ^[Yy]$ ]]
}

step "Removing systemd user units"
for unit in burnish burnish-languagetool burnish-ydotoold; do
    if [[ -e "$UNIT_DIR/$unit.service" ]]; then
        systemctl --user disable --now "$unit.service"
        rm -f "$UNIT_DIR/$unit.service"
    fi
done
systemctl --user daemon-reload

step "Removing global shortcuts"
for id in burnish-fix burnish-polish; do
    rm -f "$APP_DIR/$id.desktop"
    if command -v kwriteconfig6 >/dev/null; then
        kwriteconfig6 --file kglobalshortcutsrc --group services --group "$id.desktop" --key _launch --delete
    elif command -v kwriteconfig5 >/dev/null; then
        kwriteconfig5 --file kglobalshortcutsrc --group "$id.desktop" --key _launch --delete
        kwriteconfig5 --file kglobalshortcutsrc --group "$id.desktop" --key _k_friendly_name --delete
    fi
done
if systemctl --user is-active --quiet plasma-kglobalaccel.service; then
    systemctl --user restart plasma-kglobalaccel.service
fi

step "Removing commands and the virtual environment"
rm -f "$HOME/.local/bin/burnish" "$HOME/.local/bin/burnish-hotkey" "$HOME/.local/bin/burnish-models"
rm -rf "$REPO/.venv"

if ask "Delete model weights, LanguageTool and n-grams in ~/.local/share/burnish?"; then
    rm -rf "$HOME/.local/share/burnish"
fi
if ask "Delete the config in ~/.config/burnish and logs in ~/.local/state/burnish?"; then
    rm -rf "$HOME/.config/burnish" "$HOME/.local/state/burnish"
fi

step "Done"
cat <<'NOTE'
Not removed: system packages, and
/etc/udev/rules.d/60-burnish-uinput.rules with your 'input' group membership, if you added them.
NOTE
