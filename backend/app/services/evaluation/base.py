"""
Abstract base class for evaluators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from backend.app.domain.schemas import EvaluationReport


class Evaluator(ABC):
    """Interface for evaluating a suggested reply."""

    @abstractmethod
    def evaluate(
        self,
        incoming_email: str,
        candidate_reply: str,
        reference_reply: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Evaluate a candidate reply against the incoming email.

        Args:
            incoming_email: The original email being replied to.
            candidate_reply: The reply to evaluate.
            reference_reply: Optional ground-truth reply (holdout mode only).

        Returns:
            EvaluationReport with per-axis scores, rationale, and composite score.
        """
        ...
