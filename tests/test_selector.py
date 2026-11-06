import os
import subprocess

from burnish.config import SelectionConfig, Tier
from burnish.selector import ModelSelector, query_free_vram_mb, query_process_vram_mb

CFG = SelectionConfig(
    cache_seconds=30,
    gpu_tiers=(
        Tier("big", 60, min_free_vram_mb=12288),
        Tier("mid", 60, min_free_vram_mb=6144),
    ),
    cpu=Tier("light", 600, options={"n_gpu_layers": 0}),
)


def selector(free, loaded=0):
    return ModelSelector(CFG, lambda: loaded, lambda: free, lambda: 0.0)


def test_tiers_by_free_vram():
    assert selector(16000).select().model == "big"
    assert selector(12288).select().model == "big"
    assert selector(12287).select().model == "mid"
    assert selector(6144).select().on_gpu is True


def test_low_vram_uses_cpu_options():
    chosen = selector(2000).select()
    assert chosen.model == "light"
    assert chosen.on_gpu is False
    assert chosen.options == {"n_gpu_layers": 0}
    assert chosen.keep_alive_seconds == 600


def test_no_gpu_uses_cpu():
    assert selector(None).select().model == "light"


def test_own_loaded_model_does_not_demote_tier():
    assert selector(5000, loaded=10000).select().model == "big"


def test_result_is_cached_until_ttl():
    values = [16000, 2000]
    now = [0.0]
    chosen = ModelSelector(CFG, lambda: 0, lambda: values.pop(0), lambda: now[0])
    assert chosen.select().model == "big"
    now[0] = 29.0
    assert chosen.select().model == "big"
    assert len(values) == 1
    now[0] = 30.0
    assert chosen.select().model == "light"


def test_invalidate_forces_a_new_check():
    values = [16000, 2000]
    chosen = ModelSelector(CFG, lambda: 0, lambda: values.pop(0), lambda: 0.0)
    assert chosen.select().model == "big"
    chosen.invalidate()
    assert chosen.select().model == "light"


def test_override_keeps_device_settings():
    chosen = selector(2000).select("phi4-mini")
    assert chosen.model == "phi4-mini"
    assert chosen.options == {"n_gpu_layers": 0}


def test_query_parses_largest_gpu(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="1024\n4096\n"))
    assert query_free_vram_mb() == 4096


def test_query_handles_failures(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    def failing(*a, **k):
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    for behaviour in (missing, failing):
        monkeypatch.setattr(subprocess, "run", behaviour)
        assert query_free_vram_mb() is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="[N/A]\n"))
    assert query_free_vram_mb() is None


def test_process_vram_counts_only_this_process(monkeypatch):
    output = f"{os.getpid()}, 9000\n{os.getpid() + 1}, 500\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=output))
    assert query_process_vram_mb() == 9000


def test_process_vram_is_zero_when_unreadable(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert query_process_vram_mb() == 0
