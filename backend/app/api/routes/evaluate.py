"""
POST /api/evaluate — standalone evaluation endpoint.

Allows testing/demo of arbitrary text without going through the generation pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.domain.schemas import EvaluateRequest, EvaluateResponse

router = APIRouter()


@router.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate_reply(request: EvaluateRequest) -> EvaluateResponse:
    """Run the evaluator on an arbitrary candidate reply."""
    from backend.app.main import get_evaluator

    try:
        evaluator = get_evaluator()

        evaluation = evaluator.evaluate(
            incoming_email=request.incoming_email,
            candidate_reply=request.candidate_reply,
            reference_reply=request.reference_reply,
        )

        return EvaluateResponse(evaluation=evaluation)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
