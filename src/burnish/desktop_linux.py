import fcntl
import os
import shutil
import subprocess
from pathlib import Path

from burnish.config import HotkeyConfig
from burnish.desktop import DesktopError

_MODIFIER_KEYCODES = (29, 97, 56, 100, 42, 54, 125)
_KEY_CTRL, _KEY_C, _KEY_V = 29, 46, 47


def _run(args: list[str], stdin: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    # Clipboard writers fork a daemon that keeps inherited pipes open, so writes must not capture output.
    writes = stdin is not None
    try:
        result = subprocess.run(
            args,
            input=stdin,
            stdout=subprocess.DEVNULL if writes else subprocess.PIPE,
            stderr=subprocess.DEVNULL if writes else subprocess.PIPE,
            timeout=5,
        )
    except FileNotFoundError as e:
        raise DesktopError(f"{args[0]} is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise DesktopError(f"{args[0]} timed out") from e
    if check and result.returncode != 0:
        detail = (result.stderr or b"").decode(errors="replace").strip()
        raise DesktopError(f"{args[0]} failed: {detail}")
    return result


def _ydotool_socket() -> str:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "")
    return os.environ.get("YDOTOOL_SOCKET", f"{runtime_dir}/.ydotool_socket")


def _ydotool_ctrl(keycode: int, socket: str) -> None:
    os.environ["YDOTOOL_SOCKET"] = socket
    release = [f"{code}:0" for code in _MODIFIER_KEYCODES]
    press = [f"{_KEY_CTRL}:1", f"{keycode}:1", f"{keycode}:0", f"{_KEY_CTRL}:0"]
    _run(["ydotool", "key", *release, *press])


class X11Session:
    text_target = "UTF8_STRING"
    primary_only = False

    def __init__(self, keystrokes: str) -> None:
        self.ydotool_socket = _ydotool_socket() if keystrokes == "ydotool" else None

    def _ctrl(self, key: str, keycode: int) -> None:
        if self.ydotool_socket is None:
            _run(["xdotool", "key", "--clearmodifiers", f"ctrl+{key}"])
        elif os.path.exists(self.ydotool_socket):
            _ydotool_ctrl(keycode, self.ydotool_socket)
        else:
            raise DesktopError(f"No ydotool socket at {self.ydotool_socket}; is burnish-ydotoold running?")

    def active_window(self) -> str | None:
        return _run(["xdotool", "getactivewindow"]).stdout.decode().strip()

    def window_class(self, window: str) -> str:
        return _run(["xdotool", "getwindowclassname", window]).stdout.decode().strip()

    def _read(self, selection: str, target: str) -> bytes | None:
        result = _run(["xclip", "-selection", selection, "-o", "-t", target], check=False)
        return result.stdout if result.returncode == 0 else None

    def list_targets(self) -> list[str]:
        raw = self._read("clipboard", "TARGETS")
        return raw.decode(errors="replace").split() if raw else []

    def read_clipboard(self, target: str) -> bytes | None:
        return self._read("clipboard", target)

    def write_clipboard(self, data: bytes, target: str) -> None:
        _run(["xclip", "-selection", "clipboard", "-i", "-t", target], stdin=data)

    def clear_clipboard(self) -> None:
        _run(["xclip", "-selection", "clipboard", "-i"], stdin=b"")

    def read_primary(self) -> str:
        data = self._read("primary", "UTF8_STRING")
        return data.decode(errors="replace") if data else ""

    def send_copy(self) -> None:
        self._ctrl("c", _KEY_C)

    def send_paste(self) -> None:
        self._ctrl("v", _KEY_V)


class WaylandSession:
    # Without reliable focus information, Ctrl+C could interrupt a process in a terminal,
    # so on Wayland the selection is always read from PRIMARY.
    text_target = "text/plain;charset=utf-8"
    primary_only = True

    def __init__(self) -> None:
        self.has_kdotool = shutil.which("kdotool") is not None
        self.socket = _ydotool_socket()

    def active_window(self) -> str | None:
        return _run(["kdotool", "getactivewindow"]).stdout.decode().strip() if self.has_kdotool else None

    def window_class(self, window: str) -> str:
        return _run(["kdotool", "getwindowclassname", window]).stdout.decode().strip()

    def list_targets(self) -> list[str]:
        result = _run(["wl-paste", "--list-types"], check=False)
        return result.stdout.decode(errors="replace").split("\n") if result.returncode == 0 else []

    def read_clipboard(self, target: str) -> bytes | None:
        result = _run(["wl-paste", "-n", "-t", target], check=False)
        return result.stdout if result.returncode == 0 else None

    def write_clipboard(self, data: bytes, target: str) -> None:
        _run(["wl-copy", "-t", target], stdin=data)

    def clear_clipboard(self) -> None:
        _run(["wl-copy", "--clear"])

    def read_primary(self) -> str:
        result = _run(["wl-paste", "--primary", "-n"], check=False)
        return result.stdout.decode(errors="replace") if result.returncode == 0 else ""

    def send_copy(self) -> None:
        raise DesktopError("Copy keystrokes are not used on Wayland")

    def send_paste(self) -> None:
        # ydotool works on every compositor but needs its daemon; wtype needs no daemon but
        # fails on KWin, which doesn't expose the virtual keyboard protocol.
        if shutil.which("ydotool") and os.path.exists(self.socket):
            _ydotool_ctrl(_KEY_V, self.socket)
        else:
            _run(["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"])


def create_session(cfg: HotkeyConfig):
    kind = os.environ.get("XDG_SESSION_TYPE")
    if kind == "x11":
        return X11Session(cfg.x11_keystrokes)
    if kind == "wayland":
        return WaylandSession()
    raise DesktopError(f"Unsupported XDG_SESSION_TYPE: {kind!r}")


def notify(body: str, replace_id: str | None = None, critical: bool = False) -> str | None:
    command = ["notify-send", "-a", "Burnish", "-p", "-t", "10000" if critical else "4000"]
    if critical:
        command += ["-u", "critical"]
    if replace_id:
        command += ["-r", replace_id]
    command += ["Burnish", body]
    result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
    return result.stdout.strip() or None


def single_instance_lock(path: Path):
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle
