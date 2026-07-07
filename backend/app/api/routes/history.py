"""
GET /api/history — past suggestions from SQLite, most recent first.
"""

from __future__ import annotations

import json

from fastapi import APIRouter

from backend.app.domain.schemas import HistoryItem
from backend.app.infra import db

router = APIRouter()


@router.get("/api/history", response_model=list[HistoryItem])
async def get_history(limit: int = 50) -> list[HistoryItem]:
    """Fetch recent suggestions from the history database."""
    records = db.get_history(limit=limit)
    return [
        HistoryItem(
            id=r.id,  # type: ignore[arg-type]
            incoming_email=r.incoming_email[:200] + "..." if len(r.incoming_email) > 200 else r.incoming_email,
            generated_reply=r.generated_reply[:200] + "..." if len(r.generated_reply) > 200 else r.generated_reply,
            composite_score=r.composite_score,
            timestamp=r.timestamp,
        )
        for r in records
    ]
