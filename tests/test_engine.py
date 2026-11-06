from dataclasses import replace

import pytest

from burnish.engine import EngineError, LlamaEngine
from burnish.models import ModelStore
from burnish.selector import Selection


class FakeLlama:
    def __init__(self, reply="She goes home.", finish_reason="stop", error=None):
        self.reply = reply
        self.finish_reason = finish_reason
        self.error = error
        self.calls = []
        self.closed = False

    def create_chat_completion(self, **kwargs):
        if self.error:
            raise self.error
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.reply}, "finish_reason": self.finish_reason}]}

    def close(self):
        self.closed = True


class FakeTimer:
    def __init__(self, seconds, callback):
        self.seconds = seconds
        self.callback = callback
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def setup(cfg, tmp_path):
    custom = replace(cfg, engine=replace(cfg.engine, models_dir=tmp_path))
    for name, directory in (("gemma3:4b", "gemma3-4b"), ("qwen3:14b", "qwen3-14b")):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "m-Q4_K_M.gguf").write_bytes(b"x")

    class Setup:
        llamas = []
        loads = []
        timers = []
        next_llama = None

    def factory(**kwargs):
        Setup.loads.append(kwargs)
        llama = Setup.next_llama or FakeLlama()
        Setup.llamas.append(llama)
        return llama

    def timer(seconds, callback):
        created = FakeTimer(seconds, callback)
        Setup.timers.append(created)
        return created

    Setup.engine = LlamaEngine(custom.engine, ModelStore(custom), factory, timer)
    Setup.llamas, Setup.loads, Setup.timers = [], [], []
    return Setup


GEMMA = Selection("gemma3:4b", 60, True, options={"n_gpu_layers": -1})
GEMMA_CPU = Selection("gemma3:4b", 600, False, options={"n_gpu_layers": 0})
QWEN = Selection("qwen3:14b", 60, True, options={"n_gpu_layers": -1})


def test_chat_is_deterministic_and_passes_model_options(setup):
    reply = setup.engine.chat(GEMMA, "sys", "usr", 100)
    assert reply.text == "She goes home."
    assert reply.truncated is False
    load = setup.loads[0]
    assert load["n_gpu_layers"] == -1
    assert load["seed"] == 0
    assert load["model_path"].endswith("m-Q4_K_M.gguf")
    call = setup.llamas[0].calls[0]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 100
    assert call["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_thinking_disabled_only_for_qwen3(setup):
    setup.engine.chat(QWEN, "sys", "usr", 100)
    assert setup.llamas[-1].calls[0]["messages"][1]["content"] == "usr /no_think"
    setup.engine.chat(GEMMA, "sys", "usr", 100)
    assert setup.llamas[-1].calls[0]["messages"][1]["content"] == "usr"


def test_length_finish_is_reported_as_truncated(setup):
    setup.next_llama = FakeLlama(finish_reason="length")
    assert setup.engine.chat(GEMMA, "s", "u", 10).truncated is True


def test_model_stays_loaded_between_requests(setup):
    setup.engine.chat(GEMMA, "s", "u", 10)
    setup.engine.chat(GEMMA, "s", "u", 10)
    assert len(setup.loads) == 1
    assert setup.timers[0].cancelled is True


def test_switching_model_or_device_closes_the_previous_one(setup):
    setup.engine.chat(GEMMA, "s", "u", 10)
    first = setup.llamas[0]
    setup.engine.chat(GEMMA_CPU, "s", "u", 10)
    assert first.closed is True
    assert setup.loads[1]["n_gpu_layers"] == 0


def test_idle_timer_unloads_after_keep_alive(setup):
    setup.engine.chat(GEMMA, "s", "u", 10)
    timer = setup.timers[-1]
    assert timer.seconds == 60 and timer.started
    timer.callback()
    assert setup.llamas[0].closed is True
    setup.engine.chat(GEMMA, "s", "u", 10)
    assert len(setup.loads) == 2


def test_missing_weights_name_the_pull_command(setup):
    missing = Selection("phi4-mini", 60, True)
    with pytest.raises(EngineError, match="burnish-models pull phi4-mini"):
        setup.engine.chat(missing, "s", "u", 10)


def test_unknown_model_is_an_engine_error(setup):
    with pytest.raises(EngineError, match="Unknown model"):
        setup.engine.chat(Selection("nope", 60, True), "s", "u", 10)


def test_load_failure_is_an_engine_error(cfg, tmp_path):
    custom = replace(cfg, engine=replace(cfg.engine, models_dir=tmp_path))
    (tmp_path / "gemma3-4b").mkdir()
    (tmp_path / "gemma3-4b" / "m-Q4_K_M.gguf").write_bytes(b"x")

    def factory(**kwargs):
        raise ValueError("Failed to load model")

    engine = LlamaEngine(custom.engine, ModelStore(custom), factory, FakeTimer)
    with pytest.raises(EngineError, match="Could not load"):
        engine.chat(GEMMA, "s", "u", 10)


def test_inference_failure_is_an_engine_error_and_still_arms_the_timer(setup):
    setup.next_llama = FakeLlama(error=ValueError("context window"))
    with pytest.raises(EngineError, match="Inference"):
        setup.engine.chat(GEMMA, "s", "u", 10)
    assert setup.timers[-1].started
