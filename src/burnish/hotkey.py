import argparse
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Protocol

from loguru import logger

from burnish import client, desktop
from burnish.config import STATE_DIR, HotkeyConfig, load_config, setup_logging
from burnish.desktop import DesktopError

_PREFERRED_TARGETS = ("image/png", "UTF8_STRING", "text/plain;charset=utf-8", "text/plain", "STRING")


class Session(Protocol):
    text_target: str
    primary_only: bool

    def active_window(self) -> str | None: ...
    def window_class(self, window: str) -> str: ...
    def list_targets(self) -> list[str]: ...
    def read_clipboard(self, target: str) -> bytes | None: ...
    def write_clipboard(self, data: bytes, target: str) -> None: ...
    def clear_clipboard(self) -> None: ...
    def read_primary(self) -> str: ...
    def send_copy(self) -> None: ...
    def send_paste(self) -> None: ...


def _save_clipboard(session: Session) -> tuple[str, bytes] | None:
    targets = session.list_targets()
    for target in _PREFERRED_TARGETS:
        if target in targets:
            data = session.read_clipboard(target)
            if data is not None:
                return target, data
    return None


def _restore_clipboard(session: Session, saved: tuple[str, bytes] | None) -> None:
    if saved is None:
        session.clear_clipboard()
    else:
        session.write_clipboard(saved[1], saved[0])


def _copy_selection(session: Session, cfg: HotkeyConfig) -> str:
    # A marker rather than an empty clipboard: Klipper refills an emptied clipboard by default.
    marker = f"burnish-{uuid.uuid4().hex}".encode()
    session.write_clipboard(marker, session.text_target)
    session.send_copy()
    deadline = time.monotonic() + cfg.copy_timeout_ms / 1000
    while time.monotonic() < deadline:
        data = session.read_clipboard(session.text_target)
        if data is not None and data != marker:
            return data.decode(errors="replace")
        time.sleep(0.02)
    return ""


class Notifier:
    def __init__(self, send: Callable[..., str | None]):
        self.send = send
        self.replace_id: str | None = None

    def show(self, body: str, critical: bool = False) -> None:
        try:
            self.replace_id = self.send(body, replace_id=self.replace_id, critical=critical)
        except (OSError, subprocess.SubprocessError) as e:
            logger.error("notification failed: {}", e)


def _summary(result: dict) -> str:
    guards = [i["message"] for i in result["issues"] if i["type"] != "languagetool"]
    return "\n".join([f"Model: {result['model']}", *guards])


def run(
    session: Session,
    mode: str,
    cfg: HotkeyConfig,
    refine: Callable[[str, str], dict],
    notifier: Notifier,
    last_result_path: Path,
) -> None:
    time.sleep(cfg.settle_ms / 1000)
    window = session.active_window()
    in_terminal = window is not None and session.window_class(window).lower() in cfg.terminal_classes
    use_primary = in_terminal or session.primary_only

    saved = _save_clipboard(session)
    clipboard_touched = False
    keep_result = False
    try:
        if use_primary:
            text = session.read_primary()
        else:
            clipboard_touched = True
            text = _copy_selection(session, cfg)
        if not text.strip():
            raise DesktopError("No text selected")

        notifier.show(f"Refining ({mode})…")
        result = refine(text, mode)
        refined = result["text"]
        if refined == text:
            notifier.show(f"No changes. {_summary(result)}")
            return

        clipboard_touched = True
        session.write_clipboard(refined.encode(), session.text_target)
        if in_terminal or (window is not None and session.active_window() != window):
            keep_result = True
            where = "Terminal selection can't be replaced" if in_terminal else "Focus moved"
            notifier.show(f"{where}: result copied to the clipboard.\n{_summary(result)}")
            return

        session.send_paste()
        time.sleep(cfg.paste_delay_ms / 1000)
        if window is None:
            # Without window info a terminal can't be detected, and it would ignore the paste.
            last_result_path.write_text(refined, encoding="utf-8")
            last_result_path.chmod(0o600)
            notifier.show(f"{_summary(result)}\nNot pasted? Result saved in {last_result_path}")
        else:
            notifier.show(_summary(result))
    finally:
        if clipboard_touched and not keep_result:
            _restore_clipboard(session, saved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="burnish-hotkey")
    parser.add_argument("--mode", choices=["fix", "polish", "all"], default="all")
    args = parser.parse_args(argv)

    cfg = load_config()
    setup_logging("hotkey")
    platform = desktop.backend()
    notifier = Notifier(platform.notify)

    # A second press during a run would save our temporary clipboard as the "original".
    lock = platform.single_instance_lock(STATE_DIR / "hotkey.lock")
    if lock is None:
        notifier.show("Already running")
        return 1

    try:
        run(
            platform.create_session(cfg.hotkey),
            args.mode,
            cfg.hotkey,
            lambda text, mode: client.refine(cfg.server.url, text, mode),
            notifier,
            STATE_DIR / "last_result.txt",
        )
    except (DesktopError, client.RefineError) as e:
        logger.error(e)
        notifier.show(str(e), critical=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
