#!/usr/bin/env python3
"""
Run calibration checks: ranking, stability, and baseline comparison.

Usage:
    python backend/scripts/run_calibration.py
    python backend/scripts/run_calibration.py --output calibration_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging
from backend.app.infra.cache import ResponseCache
from backend.app.infra.groq_client import GroqClient
from backend.app.services.dataset.loader import load_dataset, load_calibration_set
from backend.app.services.dataset.splitter import split_dataset
from backend.app.services.retrieval.tfidf_retriever import TfidfRetriever
from backend.app.services.generation.groq_generator import GroqGenerator
from backend.app.services.evaluation.aggregator import PipelineEvaluator
from backend.app.services.evaluation.llm_judge import LLMJudge
from backend.app.services.calibration.calibration_runner import CalibrationRunner
from backend.app.services.calibration.stability_check import StabilityChecker
from backend.app.services.calibration.baseline_comparator import BaselineComparator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration checks")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-stability", action="store_true", help="Skip stability check (saves API calls)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline comparison (saves API calls)")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings()

    # Initialize services
    cache = ResponseCache(settings.cache_db_path)
    groq_client = GroqClient(cache=cache)
    judge = LLMJudge(groq_client=groq_client)
    evaluator = PipelineEvaluator(judge=judge)

    combined: dict = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # ── 1. Calibration ranking check ──────────────────────────────────────
    print("🎯 Running calibration ranking check...")
    cal_items = load_calibration_set(settings.calibration_path)
    runner = CalibrationRunner(evaluator)
    cal_report = runner.run(cal_items)

    print(f"\n📊 CALIBRATION RESULTS")
    print(f"  Mean good score: {cal_report.mean_good_score:.3f}")
    print(f"  Mean bad score:  {cal_report.mean_bad_score:.3f}")
    print(f"  Score gap:       {cal_report.score_gap:.3f}")
    print(f"  Good > Bad:      {'✅ YES' if cal_report.good_exceeds_bad else '❌ NO'}")
    print(f"  All correct:     {'✅ YES' if cal_report.all_correctly_ranked else '❌ NO'}")

    for item in cal_report.items:
        status = "✅" if item.correctly_ranked else "❌"
        print(f"    {status} {item.item_id} ({item.expected_label}): {item.composite_score:.2f}")

    combined["calibration"] = cal_report.model_dump()

    # ── 2. Stability check ────────────────────────────────────────────────
    if not args.skip_stability:
        print(f"\n🔄 Running stability check ({settings.stability_reruns} re-runs)...")
        first_good = next((it for it in cal_items if it.expected_label == "good"), cal_items[0])
        stability_checker = StabilityChecker(judge)
        stability_report = stability_checker.check(
            incoming_email=first_good.incoming_email,
            candidate_reply=first_good.candidate_reply,
        )

        print(f"\n📊 STABILITY RESULTS")
        print(f"  Runs:     {stability_report.runs}")
        print(f"  Scores:   {stability_report.scores}")
        print(f"  Mean:     {stability_report.mean:.3f}")
        print(f"  Variance: {stability_report.variance:.4f}")
        print(f"  Std dev:  {stability_report.std_dev:.4f}")

        combined["stability"] = stability_report.model_dump()
    else:
        print("\n⏭️  Skipping stability check")

    # ── 3. Baseline comparison ────────────────────────────────────────────
    if not args.skip_baseline:
        print("\n📏 Running baseline comparison...")
        records = load_dataset(settings.dataset_path)
        grounding, holdout = split_dataset(records)

        retriever = TfidfRetriever()
        retriever.fit(grounding)
        generator = GroqGenerator(groq_client=groq_client, cache=cache)

        sample = holdout[:3]  # Small sample to limit cost
        system_replies = []
        for record in sample:
            retrieval_result = retriever.top_k(record.incoming_email)
            generated = generator.generate(
                incoming_email=record.incoming_email,
                examples=retrieval_result.examples,
            )
            system_replies.append(generated.text)

        comparator = BaselineComparator(evaluator)
        baseline_report = comparator.compare(sample, system_replies)

        print(f"\n📊 BASELINE COMPARISON RESULTS")
        print(f"  Sample size:     {baseline_report.sample_size}")
        print(f"  Baseline mean:   {baseline_report.baseline_mean_score:.3f}")
        print(f"  System mean:     {baseline_report.system_mean_score:.3f}")
        print(f"  Delta:           {baseline_report.delta:+.3f}")
        print(f"  System better:   {'✅ YES' if baseline_report.system_better else '❌ NO'}")

        combined["baseline"] = baseline_report.model_dump()
    else:
        print("\n⏭️  Skipping baseline comparison")

    # Output
    output_path = args.output or str(settings.data_dir / "calibration_report.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n✅ Report written to {output_path}")


if __name__ == "__main__":
    main()
