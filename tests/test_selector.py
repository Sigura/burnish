import subprocess
from dataclasses import replace

import pytest

from burnish.config import SelectionConfig, Tier
from burnish.selector import LLAMA_SPLIT_MODE_NONE, ModelSelector, query_free_vram_mb

CFG = SelectionConfig(
    cache_seconds=30,
    gpu_tiers=(
        Tier("big", 60, min_free_vram_mb=12288),
        Tier("mid", 60, min_free_vram_mb=6144),
    ),
    cpu=Tier("light", 600, options={"n_gpu_layers": 0}),
)


@pytest.fixture(autouse=True)
def all_gpus_visible(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def selector(free, loaded=None, cfg=CFG):
    loaded = loaded or {}
    return ModelSelector(cfg, lambda gpu: loaded.get(gpu, 0), lambda: free, lambda: 0.0)


def test_tiers_by_free_vram():
    assert selector({0: 16000}).select().model == "big"
    assert selector({0: 12288}).select().model == "big"
    assert selector({0: 12287}).select().model == "mid"
    assert selector({0: 6144}).select().on_gpu is True


def test_low_vram_uses_cpu_options():
    chosen = selector({0: 2000}).select()
    assert chosen.model == "light"
    assert chosen.on_gpu is False
    assert chosen.options == {"n_gpu_layers": 0}
    assert chosen.keep_alive_seconds == 600


def test_no_gpu_uses_cpu():
    chosen = selector(None).select()
    assert chosen.model == "light"
    assert chosen.gpus_busy is False


def test_busy_gpus_use_cpu_and_are_flagged():
    chosen = selector({0: 2000, 1: 3000}).select()
    assert chosen.model == "light"
    assert chosen.gpus_busy is True
    assert chosen.gpu is None


def test_own_loaded_model_does_not_demote_tier():
    assert selector({0: 5000}, loaded={0: 10000}).select().model == "big"


def test_own_model_counts_only_on_its_gpu():
    chosen = selector({0: 5000, 1: 7000}, loaded={0: 10000}).select()
    assert (chosen.model, chosen.gpu) == ("big", 0)
    assert selector({0: 5000, 1: 7000}, loaded={1: 1000}).select().model == "mid"


def test_loads_on_one_gpu_that_fits():
    chosen = selector({0: 2000, 1: 13000, 2: 20000}).select()
    assert (chosen.model, chosen.gpu) == ("big", 1)
    assert chosen.options["main_gpu"] == 1
    assert chosen.options["split_mode"] == LLAMA_SPLIT_MODE_NONE


def test_higher_tier_wins_over_preference_order():
    chosen = selector({0: 7000, 1: 13000}).select()
    assert (chosen.model, chosen.gpu) == ("big", 1)


def test_only_visible_gpus_are_used_and_main_gpu_is_their_position(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,0")
    free = {0: 2000, 1: 20000, 2: 20000, 3: 13000}
    chosen = selector(free).select()
    assert (chosen.gpu, chosen.options["main_gpu"]) == (3, 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert selector(free).select().gpus_busy is True


def test_configured_gpus_set_the_preference(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    cfg = replace(CFG, gpus=(2, 0))
    chosen = selector({0: 20000, 1: 20000, 2: 20000}, cfg=cfg).select()
    assert (chosen.gpu, chosen.options["main_gpu"]) == (2, 2)
    assert selector({0: 20000, 1: 20000, 2: 1000}, cfg=cfg).select().gpu == 0
    assert selector({0: 1000, 1: 20000, 2: 1000}, cfg=cfg).select().gpus_busy is True


def test_empty_visible_list_means_no_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    chosen = selector({0: 20000}).select()
    assert chosen.model == "light"
    assert chosen.gpus_busy is False


def test_non_index_visible_devices_are_rejected(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        selector({0: 20000})


def test_result_is_cached_until_ttl():
    values = [{0: 16000}, {0: 2000}]
    now = [0.0]
    chosen = ModelSelector(CFG, lambda gpu: 0, lambda: values.pop(0), lambda: now[0])
    assert chosen.select().model == "big"
    now[0] = 29.0
    assert chosen.select().model == "big"
    assert len(values) == 1
    now[0] = 30.0
    assert chosen.select().model == "light"


def test_invalidate_forces_a_new_check():
    values = [{0: 16000}, {0: 2000}]
    chosen = ModelSelector(CFG, lambda gpu: 0, lambda: values.pop(0), lambda: 0.0)
    assert chosen.select().model == "big"
    chosen.invalidate()
    assert chosen.select().model == "light"


def test_override_keeps_device_settings():
    chosen = selector({0: 2000}).select("phi4-mini")
    assert chosen.model == "phi4-mini"
    assert chosen.options == {"n_gpu_layers": 0}


def test_query_parses_each_gpu(monkeypatch):
    output = "0, 1024\n1, 4096\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=output))
    assert query_free_vram_mb() == {0: 1024, 1: 4096}


def test_query_handles_failures(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    def failing(*a, **k):
        raise subprocess.CalledProcessError(9, "nvidia-smi")

    for behaviour in (missing, failing):
        monkeypatch.setattr(subprocess, "run", behaviour)
        assert query_free_vram_mb() is None
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="0, [N/A]\n"))
    assert query_free_vram_mb() is None
