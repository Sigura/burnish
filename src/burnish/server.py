import time
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from burnish.config import Config, load_config, setup_logging
from burnish.deepseek import DeepSeekClient, load_key
from burnish.engine import LlamaEngine
from burnish.languagetool import LanguageToolClient, LanguageToolError
from burnish.models import ModelStore
from burnish.pipeline import Pipeline, ServiceError
from burnish.selector import ModelSelector, query_free_vram_mb, query_process_vram_mb


class RefineRequest(BaseModel):
    text: str
    mode: Literal["fix", "polish", "all"] = "all"
    model: str | None = None


class RefineResponse(BaseModel):
    text: str
    diff: str
    model: str
    issues: list[dict]


def build_pipeline(cfg: Config) -> Pipeline:
    engine = LlamaEngine(cfg.engine, ModelStore(cfg))
    lt = LanguageToolClient(cfg.languagetool)
    gpu = engine.supports_gpu()
    if not gpu:
        logger.warning("llama.cpp was built without GPU support; using the CPU tier")
    free_vram = query_free_vram_mb if gpu else (lambda: None)
    key = load_key(cfg.deepseek) if cfg.deepseek.enabled else None
    remote = DeepSeekClient(cfg.deepseek, key) if key else None
    logger.info("DeepSeek: {}", remote.name if remote else "off" if not cfg.deepseek.enabled else "no key")
    return Pipeline(cfg, engine, lt, ModelSelector(cfg.selection, query_process_vram_mb, free_vram), remote)


def deepseek_status(cfg: Config, pipe: Pipeline) -> str:
    if not cfg.deepseek.enabled:
        return "off in config"
    if pipe.remote is None:
        return f"no key: set DEEPSEEK_API_KEY or write it to {cfg.deepseek.key_file}"
    status = pipe.remote.check()
    if status == "ok" and not pipe.remote_available():
        return "skipped after a failed request, retried soon"
    return status


def create_app(cfg: Config, pipeline: Pipeline | None = None) -> FastAPI:
    pipe = pipeline or build_pipeline(cfg)
    app = FastAPI(title="burnish")
    # Websites can reach localhost from the browser; DNS rebinding is blocked by the Host check.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[cfg.server.host, "127.0.0.1", "localhost"])

    @app.post("/refine", response_model=RefineResponse)
    def refine(request: RefineRequest):
        if len(request.text) > cfg.server.max_chars:
            raise HTTPException(413, f"Text is longer than {cfg.server.max_chars} characters")
        started = time.perf_counter()
        try:
            result = pipe.refine(request.text, request.mode, request.model)
        except ServiceError as e:
            logger.error("refine failed: {}", e)
            raise HTTPException(503, str(e)) from e
        logger.info(
            "refine mode={} chars={} model={} issues={} seconds={:.2f}",
            request.mode, len(request.text), result.model, len(result.issues),
            time.perf_counter() - started,
        )
        return RefineResponse(text=result.text, diff=result.diff, model=result.model, issues=result.issues)

    @app.get("/health")
    def health():
        selection = pipe.selector.select()
        report = {
            "gpu_offload": pipe.llm.supports_gpu(),
            "model": selection.model,
            "model_downloaded": pipe.llm.is_pulled(selection.model),
            "free_vram_mb": selection.free_vram_mb,
            "deepseek": deepseek_status(cfg, pipe),
            "deepseek_model": pipe.remote.name if pipe.remote else None,
        }
        try:
            pipe.lt.check("ok")
            report["languagetool"] = "ok"
        except LanguageToolError as e:
            report["languagetool"] = str(e)
        return report

    return app


def main() -> None:
    cfg = load_config()
    setup_logging("server")
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port)
