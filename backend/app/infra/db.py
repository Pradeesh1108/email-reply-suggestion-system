"""
SQLite database for persisting suggestion history and reports.

Uses SQLModel (SQLAlchemy + Pydantic) for the ORM layer — keeps the history
browseable via the /api/history endpoint and the frontend.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ORM models
# ═══════════════════════════════════════════════════════════════════════════════

class SuggestionRecord(SQLModel, table=True):
    """A persisted suggestion: incoming email + generated reply + evaluation."""
    __tablename__ = "suggestions"

    id: int | None = Field(default=None, primary_key=True)
    incoming_email: str
    generated_reply: str
    evaluation_json: str  # serialized EvaluationReport
    composite_score: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReportRecord(SQLModel, table=True):
    """Stored report (holdout, calibration, etc.)."""
    __tablename__ = "reports"

    id: int | None = Field(default=None, primary_key=True)
    report_type: str  # "holdout" | "calibration" | "stability" | "baseline"
    report_json: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# Engine / session management
# ═══════════════════════════════════════════════════════════════════════════════

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, echo=False)
        SQLModel.metadata.create_all(_engine)
        logger.info("Database initialized at %s", settings.database_url)
    return _engine


def get_session() -> Session:
    return Session(get_engine())


# ═══════════════════════════════════════════════════════════════════════════════
# Helper operations
# ═══════════════════════════════════════════════════════════════════════════════

def save_suggestion(
    incoming_email: str,
    generated_reply: str,
    evaluation_json: str,
    composite_score: float,
) -> int:
    """Persist a suggestion and return its ID."""
    with get_session() as session:
        record = SuggestionRecord(
            incoming_email=incoming_email,
            generated_reply=generated_reply,
            evaluation_json=evaluation_json,
            composite_score=composite_score,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id  # type: ignore[return-value]


def get_history(limit: int = 50) -> list[SuggestionRecord]:
    """Fetch recent suggestions, most recent first."""
    with get_session() as session:
        statement = (
            select(SuggestionRecord)
            .order_by(SuggestionRecord.id.desc())  # type: ignore[union-attr]
            .limit(limit)
        )
        return list(session.exec(statement).all())


def save_report(report_type: str, report_json: str) -> int:
    """Persist a report and return its ID."""
    with get_session() as session:
        record = ReportRecord(report_type=report_type, report_json=report_json)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id  # type: ignore[return-value]


def get_latest_report(report_type: str) -> Optional[ReportRecord]:
    """Fetch the most recent report of a given type."""
    with get_session() as session:
        statement = (
            select(ReportRecord)
            .where(ReportRecord.report_type == report_type)
            .order_by(ReportRecord.id.desc())  # type: ignore[union-attr]
            .limit(1)
        )
        return session.exec(statement).first()
