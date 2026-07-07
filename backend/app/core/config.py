"""
Application configuration via pydantic-settings.

All secrets, model names, and thresholds are loaded from environment variables
(or .env file), never hardcoded.  This is the single source of truth for every
tunable knob — changing behaviour should never require touching Python source.

Design note (cost-control §1/§6): model names are per-role so a reviewer can
swap in `openai/gpt-oss-20b` for cheaper runs without code changes.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root relative to this file's location
_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/app/core -> project root


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq API ──────────────────────────────────────────────────────────
    groq_api_key: str = ""

    # Per-role model selection (cost/quality trade-off — see README §Cost-Control)
    groq_model_generator: str = "openai/gpt-oss-120b"
    groq_model_judge: str = "openai/gpt-oss-120b"

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_top_k: int = 3
    retrieval_confidence_threshold: float = 0.15

    # ── Evaluation axis weights (must sum to 1.0) ─────────────────────────
    weight_relevance: float = 0.25
    weight_groundedness: float = 0.25
    weight_tone: float = 0.25
    weight_completeness: float = 0.25

    # Penalty per hard rule-based failure (subtracted from 5-point composite)
    rule_hard_failure_penalty: float = 1.0

    # Composite score threshold for "pass" in holdout reporting
    pass_threshold: float = 3.0

    # ── Judge stability (offline only — never on live path per §6) ────────
    stability_reruns: int = 3

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = f"sqlite:///{_PROJECT_ROOT / 'backend' / 'data' / 'history.db'}"
    cache_db_path: str = str(_PROJECT_ROOT / "backend" / "data" / "cache.db")

    # ── Paths ─────────────────────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return _PROJECT_ROOT / "backend" / "data"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "email_reply_dataset.jsonl"

    @property
    def calibration_path(self) -> Path:
        return self.data_dir / "calibration_set.jsonl"

    @property
    def axis_weights(self) -> dict[str, float]:
        return {
            "relevance_coverage": self.weight_relevance,
            "groundedness": self.weight_groundedness,
            "tone_fit": self.weight_tone,
            "completeness_actionability": self.weight_completeness,
        }


def get_settings() -> Settings:
    """Singleton-ish factory; import and call this rather than constructing Settings directly."""
    return Settings()
