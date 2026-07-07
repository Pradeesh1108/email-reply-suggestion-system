"""
FastAPI application factory.

This is the single entry point: `uvicorn backend.app.main:app` serves the
frontend as static files AND all API routes — no separate web server needed.

Startup event loads the dataset and fits the retrieval index once (cost-control
§6: index built once at startup, not per request).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.domain.schemas import EmailRecord
from backend.app.infra.cache import ResponseCache
from backend.app.infra.groq_client import GroqClient
from backend.app.services.retrieval.tfidf_retriever import TfidfRetriever
from backend.app.services.generation.groq_generator import GroqGenerator
from backend.app.services.evaluation.aggregator import PipelineEvaluator
from backend.app.services.evaluation.llm_judge import LLMJudge
from backend.app.services.dataset.loader import load_dataset
from backend.app.services.dataset.splitter import split_dataset

# ── Module-level singletons (initialized at startup) ─────────────────────────
_retriever: TfidfRetriever | None = None
_generator: GroqGenerator | None = None
_evaluator: PipelineEvaluator | None = None
_judge: LLMJudge | None = None
_records: list[EmailRecord] = []
_grounding: list[EmailRecord] = []
_holdout: list[EmailRecord] = []


def get_retriever() -> TfidfRetriever:
    assert _retriever is not None, "App not started — retriever not initialized"
    return _retriever


def get_generator() -> GroqGenerator:
    assert _generator is not None, "App not started — generator not initialized"
    return _generator


def get_evaluator() -> PipelineEvaluator:
    assert _evaluator is not None, "App not started — evaluator not initialized"
    return _evaluator


def get_judge() -> LLMJudge:
    assert _judge is not None, "App not started — judge not initialized"
    return _judge


def get_dataset_records() -> list[EmailRecord]:
    return _records


def get_split() -> tuple[list[EmailRecord], list[EmailRecord]]:
    return _grounding, _holdout


def get_settings_instance() -> Settings:
    return get_settings()


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services at startup, clean up at shutdown."""
    global _retriever, _generator, _evaluator, _judge
    global _records, _grounding, _holdout

    setup_logging()
    settings = get_settings()

    # ── Load dataset ──────────────────────────────────────────────────────
    logger.info("Loading dataset from %s", settings.dataset_path)
    _records = load_dataset(settings.dataset_path)
    logger.info("Loaded %d email/reply records", len(_records))

    # ── Split ─────────────────────────────────────────────────────────────
    _grounding, _holdout = split_dataset(_records)
    logger.info("Split: %d grounding, %d holdout", len(_grounding), len(_holdout))

    # ── Build retrieval index (once at startup — cost-control §6) ─────────
    _retriever = TfidfRetriever()
    _retriever.fit(_grounding)

    # ── Initialize Groq client with cache ─────────────────────────────────
    cache = ResponseCache(settings.cache_db_path)
    groq_client = GroqClient(cache=cache)

    # ── Initialize generator ──────────────────────────────────────────────
    _generator = GroqGenerator(groq_client=groq_client, cache=cache)

    # ── Initialize evaluator ──────────────────────────────────────────────
    _judge = LLMJudge(groq_client=groq_client)
    _evaluator = PipelineEvaluator(judge=_judge)

    logger.info("All services initialized — ready to serve")
    yield
    logger.info("Shutting down")


# ── Create app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Email Reply Suggestion System",
    description="RAG-based email assistant with tiered multi-axis evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (allow frontend dev server if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API routers ───────────────────────────────────────────────────────
from backend.app.api.routes import suggest, evaluate, dataset, holdout, calibration, history

app.include_router(suggest.router, tags=["Suggestion"])
app.include_router(evaluate.router, tags=["Evaluation"])
app.include_router(dataset.router, tags=["Dataset"])
app.include_router(holdout.router, tags=["Holdout Evaluation"])
app.include_router(calibration.router, tags=["Calibration"])
app.include_router(history.router, tags=["History"])

# ── Mount frontend static files ──────────────────────────────────────────────
_frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend index.html."""
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Frontend not found — API is still available at /docs"}
