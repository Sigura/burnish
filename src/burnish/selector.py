import os
import subprocess
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from loguru import logger

from burnish.config import SelectionConfig, Tier

# GPUs are matched to nvidia-smi indices, which follow PCI order; CUDA's default order is fastest-first.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

LLAMA_SPLIT_MODE_NONE = 0


@dataclass(frozen=True)
class Selection:
    model: str
    keep_alive_seconds: float
    on_gpu: bool
    free_vram_mb: int | None = None
    options: dict = field(default_factory=dict)
    gpu: int | None = None
    gpus_busy: bool = False
    free_ram_mb: int | None = None
    ram_short: bool = False


def query_free_vram_mb() -> dict[int, int] | None:
    command = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
        return {int(index): int(free) for index, free in rows}
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        logger.info("nvidia-smi unavailable, treating as no GPU: {}", e)
        return None


def query_free_ram_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError) as e:
        logger.info("free RAM unknown, treating as enough: {}", e)
    return None


def query_gpu_free_mb(gpu: int) -> int | None:
    free = query_free_vram_mb()
    return None if free is None else free.get(gpu)


def cuda_visible_gpus() -> list[int] | None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    try:
        return [int(item) for item in value.split(",") if item.strip()]
    except ValueError as e:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must list GPU indices, got {value!r}") from e


def candidate_gpus(free: dict[int, int], visible: list[int] | None, preferred: tuple[int, ...]) -> list[tuple[int, int]]:
    """(nvidia-smi index, position among the CUDA-visible GPUs) in order of preference."""
    order = sorted(free) if visible is None else visible
    positions = {gpu: position for position, gpu in enumerate(order)}
    return [(gpu, positions[gpu]) for gpu in preferred or order if gpu in positions and gpu in free]


def selection_from_tier(
    tier: Tier,
    on_gpu: bool,
    free_vram_mb: int | None = None,
    gpu: int | None = None,
    main_gpu: int = 0,
    gpus_busy: bool = False,
) -> Selection:
    options = dict(tier.options)
    if gpu is not None:
        options.update(main_gpu=main_gpu, split_mode=LLAMA_SPLIT_MODE_NONE)
    return Selection(tier.model, tier.keep_alive_seconds, on_gpu, free_vram_mb, options, gpu, gpus_busy)


class ModelSelector:
    def __init__(
        self,
        cfg: SelectionConfig,
        loaded_vram_mb: Callable[[int], int],
        free_vram_mb: Callable[[], dict[int, int] | None] = query_free_vram_mb,
        clock: Callable[[], float] = time.monotonic,
        free_ram_mb: Callable[[], int | None] = query_free_ram_mb,
    ):
        self.cfg = cfg
        self.loaded_vram_mb = loaded_vram_mb
        self.free_vram_mb = free_vram_mb
        self.free_ram_mb = free_ram_mb
        self.clock = clock
        self.visible = cuda_visible_gpus()
        hidden = [gpu for gpu in cfg.gpus if self.visible is not None and gpu not in self.visible]
        if hidden:
            logger.warning("[selection] gpus {} are not in CUDA_VISIBLE_DEVICES={}", hidden, self.visible)
        self._cached: Selection | None = None
        self._expires = 0.0

    def available_gpus(self) -> list[tuple[int, int, int]] | None:
        """(nvidia-smi index, CUDA position, VRAM we could use there) in order of preference."""
        free = self.free_vram_mb()
        if free is None:
            return None
        # VRAM held by our own loaded model is reusable. Without counting it, that model pushes
        # the next request into a lower tier and the two tiers keep swapping.
        return [
            (gpu, position, free[gpu] + self.loaded_vram_mb(gpu))
            for gpu, position in candidate_gpus(free, self.visible, self.cfg.gpus)
        ]

    def fit(self, tier: Tier, gpus: list[tuple[int, int, int]]) -> Selection | None:
        for gpu, position, available in gpus:
            if available >= tier.min_free_vram_mb:
                return selection_from_tier(tier, True, available, gpu, position)
        return None

    def _choose(self) -> Selection:
        gpus = self.available_gpus()
        if gpus is None:
            return self._cpu(selection_from_tier(self.cfg.cpu, False))
        for tier in self.cfg.gpu_tiers:
            chosen = self.fit(tier, gpus)
            if chosen:
                return chosen
        if gpus:
            logger.warning(
                "no GPU has room for a GPU tier; free VRAM: {}",
                ", ".join(f"GPU {gpu}: {available} MiB" for gpu, _, available in gpus),
            )
        best = max((available for _, _, available in gpus), default=None)
        return self._cpu(selection_from_tier(self.cfg.cpu, False, best, gpus_busy=bool(gpus)))

    def _cpu(self, selection: Selection) -> Selection:
        ram = self.free_ram_mb()
        return replace(selection, free_ram_mb=ram, ram_short=ram is not None and ram < self.cfg.cpu.min_free_ram_mb)

    def select(self, model: str | None = None) -> Selection:
        now = self.clock()
        if self._cached is None or now >= self._expires:
            self._cached = self._choose()
            self._expires = now + self.cfg.cache_seconds
            logger.info(
                "selected {} on GPU {} (free VRAM: {} MiB)", self._cached.model, self._cached.gpu, self._cached.free_vram_mb
            )
        return replace(self._cached, model=model) if model else self._cached

    def invalidate(self) -> None:
        self._cached = None
