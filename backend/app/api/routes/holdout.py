"""
POST /api/holdout-eval — batch evaluate the holdout split.
GET  /api/holdout-eval/latest — fetch the most recent holdout report.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.app.domain.schemas import HoldoutItemResult, HoldoutReport
from backend.app.infra import db

router = APIRouter()


@router.post("/api/holdout-eval", response_model=HoldoutReport)
async def run_holdout_eval() -> HoldoutReport:
    """Run generation + evaluation over the full holdout split."""
    from backend.app.main import get_split, get_retriever, get_generator, get_evaluator, get_settings

    try:
        settings = get_settings()
        _, holdout = get_split()
        retriever = get_retriever()
        generator = get_generator()
        evaluator = get_evaluator()

        items: list[HoldoutItemResult] = []

        for record in holdout:
            # Generate reply
            retrieval_result = retriever.top_k(record.incoming_email)
            generated = generator.generate(
                incoming_email=record.incoming_email,
                examples=retrieval_result.examples,
            )

            # Evaluate WITH reference (holdout mode)
            evaluation = evaluator.evaluate(
                incoming_email=record.incoming_email,
                candidate_reply=generated.text,
                reference_reply=record.sent_reply,
            )

            items.append(
                HoldoutItemResult(
                    record_id=record.id,
                    category=record.category,
                    incoming_email=record.incoming_email,
                    generated_reply=generated.text,
                    reference_reply=record.sent_reply,
                    evaluation=evaluation,
                )
            )

        # Aggregate
        composites = [it.evaluation.composite_score for it in items]
        axis_scores: dict[str, list[float]] = {}
        for it in items:
            if it.evaluation.judge_scores:
                for ax in it.evaluation.judge_scores.axes:
                    axis_scores.setdefault(ax.axis, []).append(ax.score)

        report = HoldoutReport(
            total_items=len(items),
            per_item=items,
            mean_composite=round(statistics.mean(composites), 3) if composites else 0.0,
            median_composite=round(statistics.median(composites), 3) if composites else 0.0,
            std_composite=round(statistics.stdev(composites), 3) if len(composites) > 1 else 0.0,
            per_axis_means={
                axis: round(statistics.mean(scores), 3)
                for axis, scores in axis_scores.items()
            },
            pass_rate=round(
                sum(1 for c in composites if c >= settings.pass_threshold) / len(composites), 3
            ) if composites else 0.0,
            pass_threshold=settings.pass_threshold,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Persist
        db.save_report("holdout", report.model_dump_json())

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/holdout-eval/latest")
async def get_latest_holdout():
    """Fetch the most recent holdout evaluation report."""
    record = db.get_latest_report("holdout")
    if not record:
        return {"status": "no_report", "message": "No holdout evaluation has been run yet."}
    return json.loads(record.report_json)
