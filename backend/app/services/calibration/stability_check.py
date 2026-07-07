"""
Judge stability check — re-runs the judge N times on the same input to
quantify score variance.

Design note (cost-control §6): this runs ONLY in calibration/offline mode,
never per live request.  It costs N extra API calls, so it's a validation
tool, not a production feature.

Design note (accuracy §1): knowing the judge's variance tells a reviewer
how much to trust a single live score — if std_dev is 0.8, a score of 3.5
really means "somewhere between 2.7 and 4.3" which is very different from
std_dev 0.1 where 3.5 means "reliably 3.4–3.6."
"""

from __future__ import annotations

import statistics
from datetime import datetime, timezone

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.domain.schemas import StabilityReport
from backend.app.services.evaluation.llm_judge import LLMJudge

logger = get_logger(__name__)


class StabilityChecker:
    """Re-runs the judge multiple times to measure score variance."""

    def __init__(self, judge: LLMJudge) -> None:
        self._judge = judge

    def check(
        self,
        incoming_email: str,
        candidate_reply: str,
        runs: int | None = None,
    ) -> StabilityReport:
        """
        Run the judge N times on the same input and report variance.

        Args:
            incoming_email: The email to judge.
            candidate_reply: The reply to judge.
            runs: Number of re-runs (defaults to STABILITY_RERUNS from config).

        Returns:
            StabilityReport with per-run scores and variance statistics.
        """
        settings = get_settings()
        n = runs or settings.stability_reruns

        scores: list[float] = []
        for i in range(n):
            logger.info("Stability run %d/%d", i + 1, n)
            judge_result = self._judge.score(
                incoming_email=incoming_email,
                candidate_reply=candidate_reply,
            )
            # Compute a simple average of the four axes as the composite
            avg = statistics.mean(a.score for a in judge_result.axes)
            scores.append(round(avg, 2))

        mean_score = statistics.mean(scores)
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        return StabilityReport(
            input_email=incoming_email,
            candidate_reply=candidate_reply,
            runs=n,
            scores=scores,
            mean=round(mean_score, 3),
            variance=round(variance, 4),
            std_dev=round(std_dev, 4),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
