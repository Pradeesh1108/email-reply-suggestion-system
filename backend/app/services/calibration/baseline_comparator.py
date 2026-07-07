"""
Baseline comparator — evaluates a naive template reply against the real system
to ensure the metric discriminates the two.

Design note (accuracy §1): if the naive baseline scores nearly as well as the
real system, the metric isn't measuring anything useful.  This check quantifies
that delta.  If the delta isn't clearly positive, the README must say so honestly.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.app.core.logging import get_logger
from backend.app.domain.schemas import BaselineComparison, EmailRecord
from backend.app.services.evaluation.base import Evaluator

logger = get_logger(__name__)

# The naive baseline reply — deliberately generic, no content from the email.
NAIVE_BASELINE_REPLY = (
    "Thank you for your email. We have received your message and will get back "
    "to you shortly. If you have any urgent concerns, please don't hesitate to "
    "reach out again.\n\nBest regards,\nSupport Team"
)


class BaselineComparator:
    """Compares the real system's scores against a naive template baseline."""

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator

    def compare(
        self,
        holdout_items: list[EmailRecord],
        system_replies: list[str],
    ) -> BaselineComparison:
        """
        Score the naive baseline and real system replies on the same emails.

        Args:
            holdout_items: The holdout email records.
            system_replies: The real system's generated replies (same order).

        Returns:
            BaselineComparison with per-item and aggregate deltas.
        """
        per_item: list[dict] = []
        baseline_scores: list[float] = []
        system_scores: list[float] = []

        for record, sys_reply in zip(holdout_items, system_replies):
            # Score the naive baseline
            baseline_eval = self._evaluator.evaluate(
                incoming_email=record.incoming_email,
                candidate_reply=NAIVE_BASELINE_REPLY,
                reference_reply=record.sent_reply,
            )

            # Score the real system's reply
            system_eval = self._evaluator.evaluate(
                incoming_email=record.incoming_email,
                candidate_reply=sys_reply,
                reference_reply=record.sent_reply,
            )

            baseline_scores.append(baseline_eval.composite_score)
            system_scores.append(system_eval.composite_score)

            per_item.append({
                "record_id": record.id,
                "baseline_score": baseline_eval.composite_score,
                "system_score": system_eval.composite_score,
                "delta": round(system_eval.composite_score - baseline_eval.composite_score, 2),
            })

            logger.info(
                "Baseline vs system for %s: %.2f vs %.2f (delta=%.2f)",
                record.id,
                baseline_eval.composite_score,
                system_eval.composite_score,
                system_eval.composite_score - baseline_eval.composite_score,
            )

        mean_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        mean_system = sum(system_scores) / len(system_scores) if system_scores else 0.0
        delta = mean_system - mean_baseline

        return BaselineComparison(
            sample_size=len(holdout_items),
            baseline_mean_score=round(mean_baseline, 3),
            system_mean_score=round(mean_system, 3),
            delta=round(delta, 3),
            system_better=delta > 0,
            per_item=per_item,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
