"""
POST /api/suggest — the primary user-facing endpoint.

Flow: incoming email → retrieval → generation (Groq call #1) → rule checks →
judge (Groq call #2, conditional) → aggregator → response.

Design note (cost-control §6): exactly 2 LLM calls max per request (1 gen + 1
judge), and the judge is skippable on hard rule failure = 1 call.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.app.domain.schemas import SuggestRequest, SuggestResponse
from backend.app.infra import db

router = APIRouter()


@router.post("/api/suggest", response_model=SuggestResponse)
async def suggest_reply(request: SuggestRequest) -> SuggestResponse:
    """Generate a suggested reply and evaluate it (reference-free mode)."""
    from backend.app.main import get_retriever, get_generator, get_evaluator

    try:
        retriever = get_retriever()
        generator = get_generator()
        evaluator = get_evaluator()

        # Step 1: Retrieve similar past examples
        retrieval_result = retriever.top_k(request.incoming_email)

        # Step 2: Generate reply (Groq call #1)
        generated = generator.generate(
            incoming_email=request.incoming_email,
            examples=retrieval_result.examples,
        )
        generated.low_retrieval_confidence = retrieval_result.low_confidence

        # Step 3: Evaluate (reference-free — no ground truth for live requests)
        evaluation = evaluator.evaluate(
            incoming_email=request.incoming_email,
            candidate_reply=generated.text,
            reference_reply=None,  # never in live mode
        )
        evaluation.retrieval_low_confidence = retrieval_result.low_confidence

        # Persist to history
        db.save_suggestion(
            incoming_email=request.incoming_email,
            generated_reply=generated.text,
            evaluation_json=evaluation.model_dump_json(),
            composite_score=evaluation.composite_score,
        )

        return SuggestResponse(
            generated_reply=generated,
            retrieved_examples=retrieval_result.examples,
            evaluation=evaluation,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
