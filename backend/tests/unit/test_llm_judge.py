"""
Unit tests for the LLM judge.

Groq client is mocked — no network access needed (spec §8).
"""

import json
import pytest
from unittest.mock import MagicMock

from backend.app.services.evaluation.llm_judge import LLMJudge
from backend.app.infra.groq_client import GroqClient


def _make_mock_client(response_json: dict) -> MagicMock:
    """Create a mock GroqClient that returns a fixed JSON response."""
    mock = MagicMock(spec=GroqClient)
    mock.chat_completion.return_value = json.dumps(response_json)
    return mock


VALID_JUDGE_RESPONSE = {
    "relevance_coverage": {"score": 4, "rationale": "Covers the main question"},
    "groundedness": {"score": 5, "rationale": "No fabricated information"},
    "tone_fit": {"score": 4, "rationale": "Appropriate formality"},
    "completeness_actionability": {"score": 3, "rationale": "Missing next steps"},
    "overall_rationale": "Good reply but could be more actionable.",
}


class TestLLMJudge:
    def test_valid_response_parsed_correctly(self):
        mock_client = _make_mock_client(VALID_JUDGE_RESPONSE)
        judge = LLMJudge(groq_client=mock_client)

        result = judge.score(
            incoming_email="I need help with my order.",
            candidate_reply="Thank you, I'll look into it.",
        )

        assert len(result.axes) == 4
        assert result.score_for("relevance_coverage") == 4
        assert result.score_for("groundedness") == 5
        assert result.score_for("tone_fit") == 4
        assert result.score_for("completeness_actionability") == 3
        assert "actionable" in result.overall_rationale.lower()

    def test_scores_clamped_to_valid_range(self):
        """Out-of-range scores should be clamped to [1, 5]."""
        response = dict(VALID_JUDGE_RESPONSE)
        response["relevance_coverage"] = {"score": 10, "rationale": "Way too high"}
        response["groundedness"] = {"score": -1, "rationale": "Way too low"}

        mock_client = _make_mock_client(response)
        judge = LLMJudge(groq_client=mock_client)

        result = judge.score("email", "reply")

        assert result.score_for("relevance_coverage") == 5  # clamped to max
        assert result.score_for("groundedness") == 1  # clamped to min

    def test_malformed_json_returns_fallback(self):
        """If the LLM returns garbage, we get fallback scores, not a crash."""
        mock_client = MagicMock(spec=GroqClient)
        mock_client.chat_completion.return_value = "This is not JSON at all!"

        judge = LLMJudge(groq_client=mock_client)
        result = judge.score("email", "reply")

        assert len(result.axes) == 4
        for axis in result.axes:
            assert axis.score == 1  # fallback minimum
            assert "parse" in axis.rationale.lower()

    def test_hallucination_flags_included_in_prompt(self):
        """Hallucination flags should be passed to the LLM call."""
        mock_client = _make_mock_client(VALID_JUDGE_RESPONSE)
        judge = LLMJudge(groq_client=mock_client)

        judge.score(
            incoming_email="Order question",
            candidate_reply="Your refund of $500 is processed.",
            hallucination_flags=["number/currency: $500"],
        )

        # Verify the flags appear in the prompt sent to the client
        call_args = mock_client.chat_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_message = messages[-1]["content"]
        assert "$500" in user_message

    def test_single_api_call_per_invocation(self):
        """Spec requirement: exactly one LLM call per judge invocation."""
        mock_client = _make_mock_client(VALID_JUDGE_RESPONSE)
        judge = LLMJudge(groq_client=mock_client)

        judge.score("email", "reply")

        assert mock_client.chat_completion.call_count == 1

    def test_score_for_missing_axis_raises(self):
        mock_client = _make_mock_client(VALID_JUDGE_RESPONSE)
        judge = LLMJudge(groq_client=mock_client)
        result = judge.score("email", "reply")

        with pytest.raises(KeyError):
            result.score_for("nonexistent_axis")
