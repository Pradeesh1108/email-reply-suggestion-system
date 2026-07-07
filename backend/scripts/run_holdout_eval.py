#!/usr/bin/env python3
"""
Run holdout evaluation — batch-evaluates the holdout split and writes a report.

Usage:
    python backend/scripts/run_holdout_eval.py
    python backend/scripts/run_holdout_eval.py --output report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging
from backend.app.domain.schemas import HoldoutItemResult, HoldoutReport
from backend.app.infra.cache import ResponseCache
from backend.app.infra.groq_client import GroqClient
from backend.app.services.dataset.loader import load_dataset
from backend.app.services.dataset.splitter import split_dataset
from backend.app.services.retrieval.tfidf_retriever import TfidfRetriever
from backend.app.services.generation.groq_generator import GroqGenerator
from backend.app.services.evaluation.aggregator import PipelineEvaluator
from backend.app.services.evaluation.llm_judge import LLMJudge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run holdout evaluation")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    # Load and split
    records = load_dataset(settings.dataset_path)
    grounding, holdout = split_dataset(records)
    print(f"📊 Dataset: {len(records)} total, {len(grounding)} grounding, {len(holdout)} holdout")

    # Initialize services
    retriever = TfidfRetriever()
    retriever.fit(grounding)

    cache = ResponseCache(settings.cache_db_path)
    groq_client = GroqClient(cache=cache)
    generator = GroqGenerator(groq_client=groq_client, cache=cache)
    judge = LLMJudge(groq_client=groq_client)
    evaluator = PipelineEvaluator(judge=judge)

    # Evaluate each holdout item
    items: list[HoldoutItemResult] = []
    for i, record in enumerate(holdout, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(holdout)}] {record.id} ({record.category})")

        retrieval_result = retriever.top_k(record.incoming_email)
        generated = generator.generate(
            incoming_email=record.incoming_email,
            examples=retrieval_result.examples,
        )

        evaluation = evaluator.evaluate(
            incoming_email=record.incoming_email,
            candidate_reply=generated.text,
            reference_reply=record.sent_reply,
        )

        items.append(HoldoutItemResult(
            record_id=record.id,
            category=record.category,
            incoming_email=record.incoming_email,
            generated_reply=generated.text,
            reference_reply=record.sent_reply,
            evaluation=evaluation,
        ))

        print(f"  Composite: {evaluation.composite_score:.2f}")
        if evaluation.judge_scores:
            for ax in evaluation.judge_scores.axes:
                print(f"    {ax.axis}: {ax.score}/5 — {ax.rationale[:60]}")

    # Aggregate
    composites = [it.evaluation.composite_score for it in items]
    axis_scores: dict[str, list[float]] = {}
    for it in items:
        if it.evaluation.judge_scores:
            for ax in it.evaluation.judge_scores.axes:
                axis_scores.setdefault(ax.axis, []).append(ax.score)

    report = HoldoutReport(
        total_items=len(items),
        per_item=items,
        mean_composite=round(statistics.mean(composites), 3) if composites else 0.0,
        median_composite=round(statistics.median(composites), 3) if composites else 0.0,
        std_composite=round(statistics.stdev(composites), 3) if len(composites) > 1 else 0.0,
        per_axis_means={
            axis: round(statistics.mean(scores), 3) for axis, scores in axis_scores.items()
        },
        pass_rate=round(
            sum(1 for c in composites if c >= settings.pass_threshold) / len(composites), 3
        ) if composites else 0.0,
        pass_threshold=settings.pass_threshold,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Output
    print(f"\n{'='*60}")
    print(f"📈 HOLDOUT EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Items evaluated: {report.total_items}")
    print(f"  Mean composite:  {report.mean_composite:.3f}")
    print(f"  Median composite:{report.median_composite:.3f}")
    print(f"  Std dev:         {report.std_composite:.3f}")
    print(f"  Pass rate (≥{report.pass_threshold}): {report.pass_rate*100:.1f}%")
    print(f"  Per-axis means:")
    for axis, mean in report.per_axis_means.items():
        print(f"    {axis}: {mean:.3f}")

    output_path = args.output or str(settings.data_dir / "holdout_report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"\n✅ Report written to {output_path}")


if __name__ == "__main__":
    main()
