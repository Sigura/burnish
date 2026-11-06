import os
import subprocess
import time
from dataclasses import dataclass, field, replace
from typing import Callable

from loguru import logger

from burnish.config import SelectionConfig, Tier


@dataclass(frozen=True)
class Selection:
    model: str
    keep_alive_seconds: float
    on_gpu: bool
    free_vram_mb: int | None = None
    options: dict = field(default_factory=dict)


def query_free_vram_mb() -> int | None:
    command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        return max(int(line) for line in result.stdout.split())
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        logger.info("nvidia-smi unavailable, treating as no GPU: {}", e)
        return None


def query_process_vram_mb() -> int:
    command = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
        rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
        return sum(int(used) for pid, used in rows if int(pid) == os.getpid())
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        logger.warning("could not read this process's VRAM use: {}", e)
        return 0


def selection_from_tier(tier: Tier, on_gpu: bool, free_vram_mb: int | None = None) -> Selection:
    return Selection(tier.model, tier.keep_alive_seconds, on_gpu, free_vram_mb, dict(tier.options))


class ModelSelector:
    def __init__(
        self,
        cfg: SelectionConfig,
        loaded_vram_mb: Callable[[], int],
        free_vram_mb: Callable[[], int | None] = query_free_vram_mb,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.loaded_vram_mb = loaded_vram_mb
        self.free_vram_mb = free_vram_mb
        self.clock = clock
        self._cached: Selection | None = None
        self._expires = 0.0

    def _choose(self) -> Selection:
        free = self.free_vram_mb()
        if free is None:
            return selection_from_tier(self.cfg.cpu, False)
        # VRAM held by our own loaded model is reusable. Without counting it, that model pushes
        # the next request into a lower tier and the two tiers keep swapping.
        available = free + self.loaded_vram_mb()
        for tier in self.cfg.gpu_tiers:
            if available >= tier.min_free_vram_mb:
                return selection_from_tier(tier, True, available)
        return selection_from_tier(self.cfg.cpu, False, available)

    def select(self, model: str | None = None) -> Selection:
        now = self.clock()
        if self._cached is None or now >= self._expires:
            self._cached = self._choose()
            self._expires = now + self.cfg.cache_seconds
            logger.info("selected {} (free VRAM: {} MiB)", self._cached.model, self._cached.free_vram_mb)
        return replace(self._cached, model=model) if model else self._cached

    def invalidate(self) -> None:
        self._cached = None
