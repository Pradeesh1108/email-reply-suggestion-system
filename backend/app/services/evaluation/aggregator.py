"""
Evaluation aggregator — combines rule-based, lexical, and judge scores.

Design note (cost-control §1): if rule-based checks find a hard failure
(placeholder leak or empty reply), the aggregator SHORT-CIRCUITS and skips
the LLM judge call entirely, saving an API call on obviously broken output.

Design note (accuracy §1): the composite score is a weighted average of the
four judge axes minus penalties for rule-based flags.  Weights are equal (0.25
each) — there's no empirical basis to weight one axis higher at this stage.
Lexical similarity is reported separately, NEVER folded into the composite.
"""

from __future__ import annotations

from typing import Optional

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.domain.schemas import EvaluationReport, RuleCheckSummary, JudgeScore
from backend.app.services.evaluation.base import Evaluator
from backend.app.services.evaluation.rule_checks import run_all_checks
from backend.app.services.evaluation.lexical_similarity import compute_lexical_similarity
from backend.app.services.evaluation.llm_judge import LLMJudge

logger = get_logger(__name__)

# Fixed low composite score assigned on hard rule failure (short-circuit).
# Rationale: 1.0 = minimum on 5-point scale, makes it unambiguously bad.
SHORT_CIRCUIT_SCORE = 1.0


class PipelineEvaluator(Evaluator):
    """
    Tiered evaluation pipeline:
    1. Rule-based checks (zero cost)
    2. Lexical similarity to reference (zero cost, when reference exists)
    3. LLM judge (one API call, skippable on hard failure)
    4. Aggregate into composite score
    """

    def __init__(self, judge: LLMJudge) -> None:
        self._judge = judge

    def evaluate(
        self,
        incoming_email: str,
        candidate_reply: str,
        reference_reply: Optional[str] = None,
    ) -> EvaluationReport:
        """Run the full evaluation pipeline."""
        settings = get_settings()

        # ── Step 1: Rule-based checks (zero cost) ────────────────────────
        rule_summary = run_all_checks(candidate_reply, incoming_email, reference_reply)

        # ── Step 2: Lexical similarity (zero cost, when reference exists) ─
        lexical_sim: Optional[float] = None
        if reference_reply is not None:
            lexical_sim = compute_lexical_similarity(candidate_reply, reference_reply)

        # ── Step 3: LLM judge (conditional) ──────────────────────────────
        if rule_summary.any_hard_failure:
            # SHORT-CIRCUIT: skip the judge call to save an API call.
            # Cost-control §1: don't spend tokens on obviously broken output.
            logger.info("Hard rule failure detected — short-circuiting judge call")
            return EvaluationReport(
                rule_checks=rule_summary,
                judge_scores=None,
                lexical_similarity=lexical_sim,
                composite_score=SHORT_CIRCUIT_SCORE,
                short_circuited=True,
            )

        # Collect hallucination flags to give as context to the judge
        hallucination_flags: list[str] = []
        for check in rule_summary.checks:
            if check.name == "naive_hallucination_flags":
                hallucination_flags = check.flagged_items

        judge_scores = self._judge.score(
            incoming_email=incoming_email,
            candidate_reply=candidate_reply,
            hallucination_flags=hallucination_flags,
        )

        # ── Step 4: Aggregate ────────────────────────────────────────────
        composite = self._compute_composite(judge_scores, rule_summary, settings)

        return EvaluationReport(
            rule_checks=rule_summary,
            judge_scores=judge_scores,
            lexical_similarity=lexical_sim,
            composite_score=round(composite, 2),
            short_circuited=False,
        )

    def _compute_composite(
        self,
        judge: JudgeScore,
        rules: RuleCheckSummary,
        settings,
    ) -> float:
        """
        Composite = weighted average of judge axes − rule penalties.

        Weights are configurable via env vars (default: equal 0.25 each).
        Penalty = RULE_HARD_FAILURE_PENALTY per soft flag (hard failures already
        short-circuited above).

        Clamped to [1.0, 5.0].
        """
        weights = settings.axis_weights
        weighted_sum = 0.0
        total_weight = 0.0

        for axis_score in judge.axes:
            w = weights.get(axis_score.axis, 0.25)
            weighted_sum += axis_score.score * w
            total_weight += w

        if total_weight > 0:
            composite = weighted_sum / total_weight
        else:
            composite = 1.0

        # Apply soft penalty for rule flags (not hard failures — those short-circuit)
        soft_flags = sum(
            1 for c in rules.checks
            if not c.passed and not c.hard_failure
        )
        penalty = soft_flags * settings.rule_hard_failure_penalty * 0.25  # scaled penalty for soft flags
        composite -= penalty

        # Clamp to [1.0, 5.0]
        return max(1.0, min(5.0, composite))
