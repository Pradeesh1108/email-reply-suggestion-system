"""
GET /api/dataset/stats — dataset statistics.
"""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter

from backend.app.domain.schemas import DatasetStats

router = APIRouter()


@router.get("/api/dataset/stats", response_model=DatasetStats)
async def dataset_stats() -> DatasetStats:
    """Return counts per category, tone distribution, split sizes."""
    from backend.app.main import get_dataset_records, get_split

    records = get_dataset_records()
    grounding, holdout = get_split()

    categories = Counter(r.category for r in records)
    tones = Counter(r.metadata.tone for r in records)
    roles = Counter(r.metadata.sender_role for r in records)
    urgencies = Counter(r.metadata.urgency for r in records)

    return DatasetStats(
        total_records=len(records),
        grounding_size=len(grounding),
        holdout_size=len(holdout),
        categories=dict(categories),
        tone_distribution=dict(tones),
        sender_role_distribution=dict(roles),
        urgency_distribution=dict(urgencies),
    )
