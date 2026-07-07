"""
Domain schemas — ALL pydantic models that cross module boundaries live here.

No bare dicts are allowed at service/API boundaries (spec §2).  Every data
contract is an explicit, validated Pydantic v2 model so structure is inspectable
and errors surface early.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class EmailMetadata(BaseModel):
    """Per-record metadata in the dataset."""
    tone: str = Field(..., pattern=r"^(formal|casual)$")
    sender_role: str = Field(..., pattern=r"^(customer|colleague|vendor)$")
    urgency: str = Field(..., pattern=r"^(low|medium|high)$")


class EmailRecord(BaseModel):
    """One (incoming_email, sent_reply) pair from the dataset."""
    id: str
    category: str
    incoming_email: str
    sent_reply: str
    metadata: EmailMetadata


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievedExample(BaseModel):
    """A past email/reply pair retrieved as a few-shot example."""
    record_id: str
    incoming_email: str
    sent_reply: str
    similarity_score: float
    category: str


class RetrievalResult(BaseModel):
    """Output of the retriever: top-k examples + confidence flag."""
    examples: list[RetrievedExample]
    low_confidence: bool = Field(
        default=False,
        description=(
            "True when the best match similarity is below the configured "
            "threshold — the suggestion is grounded in weak matches."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════════════════

class GeneratedReply(BaseModel):
    """LLM-generated suggested reply with provenance metadata."""
    text: str
    retrieved_example_ids: list[str]
    low_retrieval_confidence: bool = False
    prompt_version: str = ""
    model_used: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation — Rule-based checks
# ═══════════════════════════════════════════════════════════════════════════════

class RuleCheckResult(BaseModel):
    """Result of a single rule-based check."""
    name: str
    passed: bool
    hard_failure: bool = Field(
        default=False,
        description="If True, this failure can short-circuit and skip the LLM judge call.",
    )
    detail: str = ""
    flagged_items: list[str] = Field(default_factory=list)


class RuleCheckSummary(BaseModel):
    """Aggregate of all rule-based checks."""
    checks: list[RuleCheckResult]
    any_hard_failure: bool = False
    total_flags: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation — LLM Judge
# ═══════════════════════════════════════════════════════════════════════════════

class AxisScore(BaseModel):
    """Score for a single evaluation axis (1-5 scale)."""
    axis: str
    score: int = Field(..., ge=1, le=5)
    rationale: str


class JudgeScore(BaseModel):
    """Complete output of the LLM judge: per-axis scores + overall rationale."""
    axes: list[AxisScore]
    overall_rationale: str
    model_used: str = ""

    def score_for(self, axis_name: str) -> int:
        """Lookup score by axis name."""
        for a in self.axes:
            if a.axis == axis_name:
                return a.score
        raise KeyError(f"Axis '{axis_name}' not found in judge scores")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation — Aggregated Report
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationReport(BaseModel):
    """
    Full evaluation of a single suggested reply.

    Design note (accuracy §1): lexical_similarity is reported as a *separate,
    clearly-labeled* secondary signal — never folded silently into the composite
    score, because two very different replies can both be excellent.
    """
    rule_checks: RuleCheckSummary
    judge_scores: Optional[JudgeScore] = Field(
        default=None,
        description="None when rule-based hard failure short-circuited the judge call.",
    )
    lexical_similarity: Optional[float] = Field(
        default=None,
        description=(
            "TF-IDF cosine similarity to reference reply (holdout mode only). "
            "Secondary sanity signal — NOT included in composite score."
        ),
    )
    composite_score: float = Field(
        ...,
        description="Weighted average of judge axes minus rule-based penalties (1-5 scale).",
    )
    short_circuited: bool = Field(
        default=False,
        description="True if a hard rule failure skipped the judge call (cost-saving).",
    )
    retrieval_low_confidence: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration / Validation
# ═══════════════════════════════════════════════════════════════════════════════

class CalibrationItem(BaseModel):
    """One hand-labeled item in the calibration set."""
    id: str
    incoming_email: str
    reference_reply: str
    candidate_reply: str
    expected_label: str = Field(..., pattern=r"^(good|bad)$")
    reason: str


class CalibrationItemResult(BaseModel):
    """Evaluation result for one calibration item."""
    item_id: str
    expected_label: str
    composite_score: float
    evaluation: EvaluationReport
    correctly_ranked: bool = False


class CalibrationReport(BaseModel):
    """Results of running the evaluator over the calibration set."""
    items: list[CalibrationItemResult]
    mean_good_score: float
    mean_bad_score: float
    good_exceeds_bad: bool
    score_gap: float
    all_correctly_ranked: bool
    timestamp: str = ""


class StabilityReport(BaseModel):
    """Results of re-running the judge N times on the same input."""
    input_email: str
    candidate_reply: str
    runs: int
    scores: list[float]
    mean: float
    variance: float
    std_dev: float
    timestamp: str = ""


class BaselineComparison(BaseModel):
    """Score delta between naive baseline and real system."""
    sample_size: int
    baseline_mean_score: float
    system_mean_score: float
    delta: float
    system_better: bool
    per_item: list[dict] = Field(default_factory=list)
    timestamp: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Holdout evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class HoldoutItemResult(BaseModel):
    """Evaluation result for a single holdout item."""
    record_id: str
    category: str
    incoming_email: str
    generated_reply: str
    reference_reply: str
    evaluation: EvaluationReport


class HoldoutReport(BaseModel):
    """Aggregate report over the full holdout split."""
    total_items: int
    per_item: list[HoldoutItemResult]
    mean_composite: float
    median_composite: float
    std_composite: float
    per_axis_means: dict[str, float]
    pass_rate: float
    pass_threshold: float
    timestamp: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# API request/response models
# ═══════════════════════════════════════════════════════════════════════════════

class SuggestRequest(BaseModel):
    incoming_email: str = Field(..., min_length=10)


class SuggestResponse(BaseModel):
    generated_reply: GeneratedReply
    retrieved_examples: list[RetrievedExample]
    evaluation: EvaluationReport


class EvaluateRequest(BaseModel):
    incoming_email: str = Field(..., min_length=10)
    candidate_reply: str = Field(..., min_length=1)
    reference_reply: Optional[str] = None


class EvaluateResponse(BaseModel):
    evaluation: EvaluationReport


class DatasetStats(BaseModel):
    total_records: int
    grounding_size: int
    holdout_size: int
    categories: dict[str, int]
    tone_distribution: dict[str, int]
    sender_role_distribution: dict[str, int]
    urgency_distribution: dict[str, int]


class HistoryItem(BaseModel):
    id: int
    incoming_email: str
    generated_reply: str
    composite_score: float
    timestamp: str
