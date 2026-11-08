import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from loguru import logger

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_PATH = Path.home() / ".config" / "burnish" / "config.toml"
DATA_DIR = Path.home() / ".local" / "share" / "burnish"
STATE_DIR = Path.home() / ".local" / "state" / "burnish"


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    max_chars: int

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class EngineConfig:
    models_dir: Path
    n_ctx: int


@dataclass(frozen=True)
class ModelSource:
    repo: str
    file: str


@dataclass(frozen=True)
class LanguageToolConfig:
    url: str
    language: str
    timeout_seconds: float
    apply_categories: tuple[str, ...]
    ignore_words: tuple[str, ...]


@dataclass(frozen=True)
class DriftConfig:
    fix_min_ratio: float
    polish_min_ratio: float


@dataclass(frozen=True)
class DeepSeekConfig:
    enabled: bool
    model: str
    url: str
    timeout_seconds: float
    retry_after_seconds: float
    key_file: Path


@dataclass(frozen=True)
class Tier:
    model: str
    keep_alive_seconds: float
    candidates: tuple[str, ...] = ()
    min_free_vram_mb: int = 0
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionConfig:
    cache_seconds: float
    gpu_tiers: tuple[Tier, ...]
    cpu: Tier


@dataclass(frozen=True)
class HotkeyConfig:
    settle_ms: int
    copy_timeout_ms: int
    paste_delay_ms: int
    terminal_classes: tuple[str, ...]
    x11_keystrokes: str


@dataclass(frozen=True)
class Config:
    server: ServerConfig
    engine: EngineConfig
    models: dict[str, ModelSource]
    languagetool: LanguageToolConfig
    drift: DriftConfig
    selection: SelectionConfig
    hotkey: HotkeyConfig
    deepseek: DeepSeekConfig

    def all_models(self) -> list[str]:
        tiers = (*self.selection.gpu_tiers, self.selection.cpu)
        names = [name for tier in tiers for name in (tier.model, *tier.candidates)]
        return list(dict.fromkeys(names))


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _tier(raw: dict) -> Tier:
    return Tier(**{**raw, "candidates": tuple(raw.get("candidates", ()))})


def _build(raw: dict) -> Config:
    lt = raw["languagetool"]
    selection = raw["selection"]
    hotkey = raw["hotkey"]
    engine = raw["engine"]
    models = {name: ModelSource(**source) for name, source in raw["models"].items()}
    tiers = sorted((_tier(t) for t in selection["gpu_tiers"]), key=lambda t: -t.min_free_vram_mb)
    cfg = Config(
        server=ServerConfig(**raw["server"]),
        engine=EngineConfig(
            models_dir=Path(engine["models_dir"]).expanduser() if engine["models_dir"] else DATA_DIR / "models",
            n_ctx=engine["n_ctx"],
        ),
        models=models,
        languagetool=LanguageToolConfig(
            **{
                **lt,
                "apply_categories": tuple(lt["apply_categories"]),
                "ignore_words": tuple(lt["ignore_words"]),
            }
        ),
        drift=DriftConfig(**raw["drift"]),
        selection=SelectionConfig(
            cache_seconds=selection["cache_seconds"],
            gpu_tiers=tuple(tiers),
            cpu=_tier(selection["cpu"]),
        ),
        hotkey=HotkeyConfig(**{**hotkey, "terminal_classes": tuple(hotkey["terminal_classes"])}),
        deepseek=DeepSeekConfig(**{**raw["deepseek"], "key_file": Path(raw["deepseek"]["key_file"]).expanduser()}),
    )
    unknown = set(cfg.all_models()) - set(models)
    if unknown:
        raise ValueError(f"Models used in [selection] have no [models] entry: {sorted(unknown)}")
    if cfg.hotkey.x11_keystrokes not in ("xdotool", "ydotool"):
        raise ValueError(f"[hotkey] x11_keystrokes must be xdotool or ydotool, got {cfg.hotkey.x11_keystrokes!r}")
    return cfg


def load_config(path: Path = CONFIG_PATH) -> Config:
    defaults = tomllib.loads(resources.files("burnish").joinpath("default_config.toml").read_text(encoding="utf-8"))
    user = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _build(_merge(defaults, user))


def setup_logging(name: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(STATE_DIR / f"{name}.log", rotation="5 MB", retention=3)
