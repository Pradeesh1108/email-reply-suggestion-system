"""
Integration test: calibration sanity check.

Spec requirement §8: asserts the mean score of good-labeled calibration items
exceeds the mean score of bad-labeled items — the metric's own sanity check,
codified as a test.

This test uses mocked judge scores (no real API key needed) but exercises the
full evaluation pipeline including rule checks and aggregation.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from backend.app.domain.schemas import AxisScore, CalibrationItem, JudgeScore
from backend.app.services.evaluation.aggregator import PipelineEvaluator
from backend.app.services.evaluation.llm_judge import LLMJudge
from backend.app.services.calibration.calibration_runner import CalibrationRunner


def _load_calibration_items() -> list[CalibrationItem]:
    """Load the actual calibration set from the data directory."""
    cal_path = Path(__file__).resolve().parents[2] / "data" / "calibration_set.jsonl"
    items = []
    with open(cal_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(CalibrationItem.model_validate(json.loads(line)))
    return items


def _make_smart_mock_judge() -> MagicMock:
    """
    Create a mock judge that returns different scores based on reply quality.

    The mock gives high scores to substantive replies and low scores to
    obviously bad ones (empty, off-topic, overly verbose).  This simulates
    what a real LLM judge would do without making API calls.
    """
    def score_fn(incoming_email: str, candidate_reply: str, hallucination_flags=None):
        reply = candidate_reply.strip()

        # Empty or very short → low scores
        if len(reply) < 10:
            return _make_judge_score(1, 1, 1, 1, "Empty or near-empty reply")

        # Has placeholders → low scores
        if "[Your Name]" in reply or "{{" in reply:
            return _make_judge_score(1, 1, 1, 1, "Unresolved template placeholders")

        # Check relevance: does it reference anything from the email?
        email_words = set(incoming_email.lower().split())
        reply_words = set(reply.lower().split())
        overlap = len(email_words & reply_words)

        # Very long, pompous reply → lower scores
        if len(reply) > 2000:
            return _make_judge_score(2, 3, 1, 2, "Excessively long reply")

        # Decent overlap = relevant reply
        if overlap > 5:
            # Check for hallucination flags
            flag_penalty = min(2, len(hallucination_flags)) if hallucination_flags else 0
            base = 4
            return _make_judge_score(
                base, max(1, base - flag_penalty), base, base,
                "Relevant and grounded reply"
            )

        # Low overlap = probably off-topic
        return _make_judge_score(2, 3, 3, 2, "Limited relevance to the email")

    mock = MagicMock(spec=LLMJudge)
    mock.score.side_effect = score_fn
    return mock


def _make_judge_score(rel: int, gnd: int, tone: int, comp: int, rationale: str) -> JudgeScore:
    return JudgeScore(
        axes=[
            AxisScore(axis="relevance_coverage", score=rel, rationale=f"Relevance: {rationale}"),
            AxisScore(axis="groundedness", score=gnd, rationale=f"Groundedness: {rationale}"),
            AxisScore(axis="tone_fit", score=tone, rationale=f"Tone: {rationale}"),
            AxisScore(axis="completeness_actionability", score=comp, rationale=f"Completeness: {rationale}"),
        ],
        overall_rationale=rationale,
        model_used="mock-judge",
    )


class TestCalibrationSanity:
    def test_good_items_outscore_bad_items(self):
        """
        The metric's sanity check: good-labeled calibration items must have a
        higher mean composite score than bad-labeled items.

        This validates that the evaluation pipeline actually discriminates
        quality — it's not just producing arbitrary numbers.
        """
        cal_items = _load_calibration_items()
        assert len(cal_items) >= 5, "Need at least 5 calibration items"

        mock_judge = _make_smart_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)
        runner = CalibrationRunner(evaluator)

        report = runner.run(cal_items)

        # Key assertion: good items must outscore bad items
        assert report.good_exceeds_bad, (
            f"Calibration FAILED: mean good ({report.mean_good_score:.2f}) "
            f"does not exceed mean bad ({report.mean_bad_score:.2f}). "
            f"The metric is not discriminating quality."
        )

        # Bonus: the gap should be meaningful (at least 0.5 on a 5-point scale)
        assert report.score_gap >= 0.5, (
            f"Score gap too small ({report.score_gap:.2f}). "
            f"The metric barely discriminates good from bad."
        )

    def test_placeholder_reply_scores_minimum(self):
        """A reply with template placeholders should get the minimum score."""
        mock_judge = _make_smart_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        report = evaluator.evaluate(
            incoming_email="I need help with my account.",
            candidate_reply="Dear [Customer Name], your {{product}} is ready.",
        )

        # Should be short-circuited with minimum score
        assert report.short_circuited is True
        assert report.composite_score == 1.0

    def test_empty_reply_scores_minimum(self):
        """An empty reply should get the minimum score."""
        mock_judge = _make_smart_mock_judge()
        evaluator = PipelineEvaluator(judge=mock_judge)

        report = evaluator.evaluate(
            incoming_email="I need help with my account.",
            candidate_reply="",
        )

        assert report.short_circuited is True
        assert report.composite_score == 1.0
