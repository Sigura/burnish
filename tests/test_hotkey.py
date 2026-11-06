from dataclasses import replace

import pytest

from burnish.desktop import DesktopError
from burnish.hotkey import Notifier, run


class FakeSession:
    text_target = "UTF8_STRING"
    primary_only = False

    def __init__(self, window="1", window_class="TelegramDesktop", selection="She go home.", primary=""):
        self.window = window
        self.klass = window_class
        self.selection = selection
        self.primary = primary
        self.clipboard = {"UTF8_STRING": b"original"}
        self.pasted = []
        self.copies = 0
        self.after_refine_window = window

    def active_window(self):
        return self.window

    def window_class(self, window):
        return self.klass

    def list_targets(self):
        return list(self.clipboard)

    def read_clipboard(self, target):
        return self.clipboard.get(target)

    def write_clipboard(self, data, target):
        self.clipboard = {target: data}

    def clear_clipboard(self):
        self.clipboard = {}

    def read_primary(self):
        return self.primary

    def send_copy(self):
        self.copies += 1
        if self.selection:
            self.clipboard = {"UTF8_STRING": self.selection.encode()}

    def send_paste(self):
        self.pasted.append(self.clipboard[self.text_target].decode())


@pytest.fixture
def hk(cfg):
    return replace(cfg.hotkey, settle_ms=0, copy_timeout_ms=50, paste_delay_ms=0)


def go(session, hk, tmp_path, refined="She goes home."):
    messages = []

    def send(body, replace_id=None, critical=False):
        messages.append(body)
        return "1"

    def refine(text, mode):
        session.window = session.after_refine_window
        return {"text": refined, "diff": "", "model": "m", "issues": []}

    run(session, "fix", hk, refine, Notifier(send), tmp_path / "last.txt")
    return messages


def test_pastes_result_and_restores_clipboard(hk, tmp_path):
    session = FakeSession()
    messages = go(session, hk, tmp_path)
    assert session.pasted == ["She goes home."]
    assert session.clipboard == {"UTF8_STRING": b"original"}
    assert "Model: m" in messages[-1]


def test_rich_clipboard_is_restored_with_its_type(hk, tmp_path):
    session = FakeSession()
    session.clipboard = {"image/png": b"\x89PNG"}
    go(session, hk, tmp_path)
    assert session.clipboard == {"image/png": b"\x89PNG"}


def test_empty_clipboard_is_cleared_again(hk, tmp_path):
    session = FakeSession()
    session.clipboard = {}
    go(session, hk, tmp_path)
    assert session.clipboard == {}


def test_no_selection_errors_and_restores(hk, tmp_path):
    session = FakeSession(selection="")
    with pytest.raises(DesktopError, match="No text selected"):
        go(session, hk, tmp_path)
    assert session.clipboard == {"UTF8_STRING": b"original"}
    assert session.pasted == []


def test_backend_failure_restores_clipboard(hk, tmp_path):
    session = FakeSession()

    def refine(text, mode):
        raise RuntimeError("backend down")

    with pytest.raises(RuntimeError):
        run(session, "fix", hk, refine, Notifier(lambda *a, **k: "1"), tmp_path / "last.txt")
    assert session.clipboard == {"UTF8_STRING": b"original"}


def test_terminal_never_gets_keystrokes_and_keeps_result(hk, tmp_path):
    session = FakeSession(window_class="konsole", primary="She go home.")
    messages = go(session, hk, tmp_path)
    assert session.copies == 0
    assert session.pasted == []
    assert session.clipboard == {"UTF8_STRING": b"She goes home."}
    assert "copied to the clipboard" in messages[-1]


def test_focus_change_copies_instead_of_pasting(hk, tmp_path):
    session = FakeSession()
    session.after_refine_window = "2"
    messages = go(session, hk, tmp_path)
    assert session.pasted == []
    assert session.clipboard == {"UTF8_STRING": b"She goes home."}
    assert "Focus moved" in messages[-1]


def test_unchanged_text_is_not_pasted(hk, tmp_path):
    session = FakeSession()
    messages = go(session, hk, tmp_path, refined="She go home.")
    assert session.pasted == []
    assert session.clipboard == {"UTF8_STRING": b"original"}
    assert "No changes" in messages[-1]


def test_primary_only_session_without_window_info(hk, tmp_path):
    session = FakeSession(window=None, primary="She go home.")
    session.primary_only = True
    session.text_target = "text/plain;charset=utf-8"
    session.clipboard = {session.text_target: b"original"}
    messages = go(session, hk, tmp_path)
    assert session.copies == 0
    assert session.pasted == ["She goes home."]
    assert session.clipboard == {session.text_target: b"original"}
    assert (tmp_path / "last.txt").read_text() == "She goes home."
    assert "last.txt" in messages[-1]
