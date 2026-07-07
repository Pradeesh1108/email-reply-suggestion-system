"""
GET  /api/calibration-report — fetch latest calibration results.
POST /api/calibration-report/run — run calibration + stability + baseline checks.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.app.infra import db

router = APIRouter()


@router.post("/api/calibration-report/run")
async def run_calibration():
    """Run calibration ranking check, stability check, and baseline comparison."""
    from backend.app.main import (
        get_evaluator, get_judge, get_retriever, get_generator, get_split,
    )
    from backend.app.services.calibration.calibration_runner import CalibrationRunner
    from backend.app.services.calibration.stability_check import StabilityChecker
    from backend.app.services.calibration.baseline_comparator import BaselineComparator
    from backend.app.services.dataset.loader import load_calibration_set
    from backend.app.core.config import get_settings

    try:
        settings = get_settings()
        evaluator = get_evaluator()
        judge = get_judge()

        # ── 1. Calibration ranking check ──────────────────────────────────
        cal_items = load_calibration_set(settings.calibration_path)
        cal_runner = CalibrationRunner(evaluator)
        cal_report = cal_runner.run(cal_items)

        # ── 2. Stability check ────────────────────────────────────────────
        # Use the first good calibration item as the fixed input
        first_good = next(
            (it for it in cal_items if it.expected_label == "good"),
            cal_items[0],
        )
        stability_checker = StabilityChecker(judge)
        stability_report = stability_checker.check(
            incoming_email=first_good.incoming_email,
            candidate_reply=first_good.candidate_reply,
        )

        # ── 3. Baseline comparison ────────────────────────────────────────
        _, holdout = get_split()
        retriever = get_retriever()
        generator = get_generator()

        # Generate system replies for a sample of holdout items
        sample = holdout[:5]  # first 5 to limit cost
        system_replies = []
        for record in sample:
            retrieval_result = retriever.top_k(record.incoming_email)
            generated = generator.generate(
                incoming_email=record.incoming_email,
                examples=retrieval_result.examples,
            )
            system_replies.append(generated.text)

        comparator = BaselineComparator(evaluator)
        baseline_report = comparator.compare(sample, system_replies)

        # ── Combine and persist ───────────────────────────────────────────
        combined = {
            "calibration": cal_report.model_dump(),
            "stability": stability_report.model_dump(),
            "baseline": baseline_report.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        db.save_report("calibration", json.dumps(combined))

        return combined

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/calibration-report")
async def get_calibration_report():
    """Fetch the most recent calibration report."""
    record = db.get_latest_report("calibration")
    if not record:
        return {"status": "no_report", "message": "No calibration report has been generated yet."}
    return json.loads(record.report_json)
