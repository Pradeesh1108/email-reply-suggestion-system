"""
LLM judge — the ONE expensive evaluation call per response.

Design note (cost-control §6): all four axes are scored in a SINGLE structured
JSON response.  This is NOT one call per axis — batching them saves 3 API calls
per evaluation.  The judge call is skippable entirely when rule-based checks
detect a hard failure (see aggregator.py).

Design note (accuracy §1): the judge is prompted to reason about hallucination
flags from the rule-based checks, but use its own judgment — the regex heuristic
produces false positives, so the judge provides the nuanced interpretation.
"""

from __future__ import annotations

import json
from typing import Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.domain.schemas import AxisScore, JudgeScore
from backend.app.infra.groq_client import GroqClient
from backend.app.services.generation.prompt_templates import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    format_hallucination_context,
)

logger = get_logger(__name__)


class LLMJudge:
    """Scores a candidate reply on four axes via a single Groq call."""

    AXES = [
        "relevance_coverage",
        "groundedness",
        "tone_fit",
        "completeness_actionability",
    ]

    def __init__(self, groq_client: GroqClient) -> None:
        self._client = groq_client

    def score(
        self,
        incoming_email: str,
        candidate_reply: str,
        hallucination_flags: Optional[list[str]] = None,
    ) -> JudgeScore:
        """
        Score a candidate reply on all four axes in one LLM call.

        Args:
            incoming_email: The original email.
            candidate_reply: The reply to judge.
            hallucination_flags: Items flagged by rule_checks (context for the judge).

        Returns:
            JudgeScore with per-axis scores, rationales, and overall rationale.
        """
        settings = get_settings()

        hallucination_context = format_hallucination_context(hallucination_flags or [])

        user_message = JUDGE_USER_TEMPLATE.format(
            incoming_email=incoming_email,
            candidate_reply=candidate_reply,
            hallucination_context=hallucination_context,
        )

        raw_response = self._client.chat_completion(
            model=settings.groq_model_judge,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,  # low temp for consistent scoring
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        return self._parse_response(raw_response, settings.groq_model_judge)

    def _parse_response(self, raw: str, model_used: str) -> JudgeScore:
        """Parse the JSON response from the judge into a JudgeScore."""
        try:
            # Strip potential markdown code blocks
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse judge response as JSON: %s", e)
            logger.debug("Raw response: %s", raw)
            # Return minimum scores with error rationale
            return self._fallback_score(
                f"JSON parse error: {e}", model_used
            )

        axes: list[AxisScore] = []
        for axis_name in self.AXES:
            axis_data = data.get(axis_name, {})
            if isinstance(axis_data, dict):
                score = int(axis_data.get("score", 1))
                rationale = str(axis_data.get("rationale", "No rationale provided"))
            else:
                score = 1
                rationale = f"Unexpected format for axis {axis_name}"

            # Clamp to 1-5
            score = max(1, min(5, score))

            axes.append(AxisScore(axis=axis_name, score=score, rationale=rationale))

        overall_rationale = data.get("overall_rationale", "No overall rationale provided")

        return JudgeScore(
            axes=axes,
            overall_rationale=str(overall_rationale),
            model_used=model_used,
        )

    def _fallback_score(self, error_detail: str, model_used: str) -> JudgeScore:
        """Return minimum scores when the judge response can't be parsed."""
        return JudgeScore(
            axes=[
                AxisScore(
                    axis=name,
                    score=1,
                    rationale=f"Could not parse judge response: {error_detail}",
                )
                for name in self.AXES
            ],
            overall_rationale=f"Judge response parse failure: {error_detail}",
            model_used=model_used,
        )
