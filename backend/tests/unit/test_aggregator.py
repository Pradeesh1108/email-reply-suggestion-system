"""
Unit tests for the evaluation aggregator.

Spec requirement §8: must include a test that a rule-based hard failure
short-circuits and skips the judge call (assert the mock judge was NOT called).

Runs fully offline — Groq client is mocked.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.app.domain.schemas import AxisScore, JudgeScore
from backend.app.services.evaluation.aggregator import PipelineEvaluator, SHORT_CIRCUIT_SCORE
from backend.app.services.evaluation.llm_judge import LLMJudge


def _make_mock_judge() -> MagicMock:
    """Create a mock judge that returns a fixed score."""
    mock = MagicMock(spec=LLMJudge)
    mock.score.return_value = JudgeScore(
        axes=[
            AxisScore(axis="relevance_coverage", score=4, rationale="Good coverage"),
            AxisScore(axis="groundedness", score=4, rationale="Well grounded"),
            AxisScore(axis="tone_fit", score=4, rationale="Appropriate tone"),
            AxisScore(axis="completeness_actionability", score=4, rationale="Complete"),
        ],
        overall_rationale="Solid reply overall.",
        model_used="test-model",
    )
    return mock


class TestPipelineEvaluator:
    def test_normal_evaluation_calls_judge(self):
        """A clean reply should invoke the judge."""
        mock_judge = _make_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        report = evaluator.evaluate(
            incoming_email="I need help with my order.",
            candidate_reply="Thank you for reaching out. I'll look into your order right away.",
        )

        assert mock_judge.score.called
        assert report.judge_scores is not None
        assert report.short_circuited is False
        assert report.composite_score > 0

    def test_hard_failure_short_circuits_judge(self):
        """
        Spec requirement §8: a rule-based hard failure must skip the judge call.

        This is the key cost-saving behavior: if the reply has a placeholder
        leak or is empty, we don't spend an API call on the judge.
        """
        mock_judge = _make_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        # Reply with placeholder → hard failure
        report = evaluator.evaluate(
            incoming_email="I need help with my order.",
            candidate_reply="Dear [Your Name], thank you for contacting {{company_name}}.",
        )

        # Assert judge was NOT called (the key assertion per spec §8)
        assert not mock_judge.score.called, "Judge should NOT be called on hard rule failure"
        assert report.short_circuited is True
        assert report.judge_scores is None
        assert report.composite_score == SHORT_CIRCUIT_SCORE

    def test_empty_reply_short_circuits(self):
        """Empty replies should also short-circuit."""
        mock_judge = _make_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        report = evaluator.evaluate(
            incoming_email="I need help.",
            candidate_reply="",
        )

        assert not mock_judge.score.called
        assert report.short_circuited is True
        assert report.composite_score == SHORT_CIRCUIT_SCORE

    def test_composite_score_in_valid_range(self):
        """Composite score should be clamped to [1.0, 5.0]."""
        mock_judge = _make_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        report = evaluator.evaluate(
            incoming_email="I need help with my order.",
            candidate_reply="Thank you for reaching out. I'll look into your order right away and process a refund.",
        )

        assert 1.0 <= report.composite_score <= 5.0

    def test_lexical_similarity_computed_when_reference_exists(self):
        """When a reference reply is provided, lexical similarity should be computed."""
        mock_judge = _make_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        report = evaluator.evaluate(
            incoming_email="I need help.",
            candidate_reply="Thank you for reaching out. I'll help you right away.",
            reference_reply="Thanks for contacting us. I'll look into this immediately.",
        )

        assert report.lexical_similarity is not None
        assert 0.0 <= report.lexical_similarity <= 1.0

    def test_no_lexical_similarity_without_reference(self):
        """Without a reference reply, lexical similarity should be None."""
        mock_judge = _make_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        report = evaluator.evaluate(
            incoming_email="I need help.",
            candidate_reply="Thank you for reaching out. I'll help you right away.",
        )

        assert report.lexical_similarity is None

    def test_hallucination_flags_passed_to_judge(self):
        """Hallucination flags should be passed as context to the judge."""
        mock_judge = _make_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        evaluator.evaluate(
            incoming_email="I was charged $29 for my subscription.",
            candidate_reply="I've processed a refund of $500 to your account immediately.",
        )

        # Judge should be called with hallucination flags
        assert mock_judge.score.called
        call_kwargs = mock_judge.score.call_args
        # The hallucination_flags kwarg should contain the flagged amount
        flags = call_kwargs.kwargs.get("hallucination_flags") or call_kwargs[1].get("hallucination_flags", [])
        assert any("$500" in flag for flag in flags)
