import gc
import threading
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from burnish.config import EngineConfig
from burnish.models import ModelError, ModelStore
from burnish.selector import Selection


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    truncated: bool


def _create_llama(**kwargs) -> Any:
    from llama_cpp import Llama

    llama = Llama(**kwargs)
    formatter = no_think_formatter(llama)
    if formatter is not None:
        llama.chat_handler = formatter.to_chat_handler()
    return llama


def no_think_formatter(llama: Any) -> Any:
    # Qwen3.8's template opens <think> unless enable_thinking=False and ignores /no_think.
    # create_chat_completion can't pass template variables, so the default handler is replaced.
    template = llama.metadata.get("tokenizer.chat_template", "")
    if "enable_thinking" not in template:
        return None
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    class NoThinkFormatter(Jinja2ChatFormatter):
        def __call__(self, **kwargs):
            return super().__call__(enable_thinking=False, **kwargs)

    def text(token: int) -> str:
        return llama.detokenize([token], special=True).decode("utf-8", "replace") if token != -1 else ""

    eos = llama.token_eos()
    return NoThinkFormatter(template=template, eos_token=text(eos), bos_token=text(llama.token_bos()), stop_token_ids=[eos])


def supports_gpu_offload() -> bool:
    from llama_cpp import llama_supports_gpu_offload

    return bool(llama_supports_gpu_offload())


class LlamaEngine:
    def __init__(
        self,
        cfg: EngineConfig,
        store: ModelStore,
        factory: Callable[..., Any] = _create_llama,
        timer_factory: Callable[..., Any] = threading.Timer,
    ):
        self.cfg = cfg
        self.store = store
        self.factory = factory
        self.timer_factory = timer_factory
        # llama.cpp contexts aren't thread-safe, and the server handles requests on a thread pool.
        self._lock = threading.RLock()
        self._llama: Any = None
        self._key: tuple | None = None
        self._timer: Any = None

    def supports_gpu(self) -> bool:
        return supports_gpu_offload()

    def is_pulled(self, model: str) -> bool:
        try:
            return self.store.path(model) is not None
        except ModelError:
            return False

    def installed(self) -> set[str]:
        return {name for name in self.store.sources if self.is_pulled(name)}

    def _ensure_loaded(self, selection: Selection) -> Any:
        key = (selection.model, tuple(sorted(selection.options.items())))
        if key == self._key:
            return self._llama
        self.unload()
        try:
            path = self.store.path(selection.model)
        except ModelError as e:
            raise EngineError(str(e)) from e
        if path is None:
            raise EngineError(f"Model {selection.model} is not downloaded. Run: burnish-models pull {selection.model}")
        logger.info("loading {} with {}", selection.model, selection.options)
        try:
            self._llama = self.factory(
                model_path=str(path), n_ctx=self.cfg.n_ctx, seed=0, verbose=False, **selection.options
            )
        except (ValueError, RuntimeError, OSError) as e:
            raise EngineError(f"Could not load {selection.model}: {e}") from e
        self._key = key
        return self._llama

    def chat(self, selection: Selection, system: str, user: str, max_tokens: int) -> Reply:
        if selection.model.startswith("qwen3"):
            user = f"{user} /no_think"
        with self._lock:
            self._cancel_timer()
            llama = self._ensure_loaded(selection)
            try:
                response = llama.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    repeat_penalty=1.0,
                    seed=0,
                    max_tokens=max_tokens,
                )
            except (ValueError, RuntimeError) as e:
                raise EngineError(f"Inference with {selection.model} failed: {e}") from e
            finally:
                self._start_timer(selection.keep_alive_seconds)
        choice = response["choices"][0]
        return Reply(choice["message"]["content"] or "", choice["finish_reason"] == "length")

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _start_timer(self, seconds: float) -> None:
        self._timer = self.timer_factory(seconds, self.unload)
        self._timer.daemon = True
        self._timer.start()

    def unload(self) -> None:
        with self._lock:
            self._cancel_timer()
            if self._llama is None:
                return
            logger.info("unloading {}", self._key[0])
            self._llama.close()
            self._llama = None
            self._key = None
            gc.collect()
