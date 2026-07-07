"""
Calibration runner — validates that the scoring metric actually measures
something real by checking that hand-labeled good items outscore bad items.

Design note (accuracy §1): a score with no validation is just an arbitrary
number.  This is the sanity test that the evaluator discriminates quality.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.core.logging import get_logger
from backend.app.domain.schemas import (
    CalibrationItem,
    CalibrationItemResult,
    CalibrationReport,
)
from backend.app.services.evaluation.base import Evaluator

logger = get_logger(__name__)


class CalibrationRunner:
    """Runs the evaluator over the calibration set and checks ranking."""

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator

    def run(self, calibration_items: list[CalibrationItem]) -> CalibrationReport:
        """
        Evaluate every calibration item and check ranking correctness.

        Returns:
            CalibrationReport with per-item results and ranking metrics.
        """
        results: list[CalibrationItemResult] = []

        for item in calibration_items:
            logger.info("Calibrating item %s (expected: %s)", item.id, item.expected_label)

            evaluation = self._evaluator.evaluate(
                incoming_email=item.incoming_email,
                candidate_reply=item.candidate_reply,
                reference_reply=item.reference_reply,
            )

            results.append(
                CalibrationItemResult(
                    item_id=item.id,
                    expected_label=item.expected_label,
                    composite_score=evaluation.composite_score,
                    evaluation=evaluation,
                )
            )

        # Compute group means
        good_scores = [r.composite_score for r in results if r.expected_label == "good"]
        bad_scores = [r.composite_score for r in results if r.expected_label == "bad"]

        mean_good = sum(good_scores) / len(good_scores) if good_scores else 0.0
        mean_bad = sum(bad_scores) / len(bad_scores) if bad_scores else 0.0

        # Check ranking: every good item should score higher than every bad item
        # (ideal), but at minimum mean(good) > mean(bad)
        good_exceeds_bad = mean_good > mean_bad
        score_gap = mean_good - mean_bad

        # Check individual ranking: does each good item score above the
        # median bad score, and vice versa?
        if bad_scores and good_scores:
            median_bad = sorted(bad_scores)[len(bad_scores) // 2]
            median_good = sorted(good_scores)[len(good_scores) // 2]
            for r in results:
                if r.expected_label == "good":
                    r.correctly_ranked = r.composite_score > median_bad
                else:
                    r.correctly_ranked = r.composite_score < median_good

        all_correct = all(r.correctly_ranked for r in results)

        return CalibrationReport(
            items=results,
            mean_good_score=round(mean_good, 3),
            mean_bad_score=round(mean_bad, 3),
            good_exceeds_bad=good_exceeds_bad,
            score_gap=round(score_gap, 3),
            all_correctly_ranked=all_correct,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
