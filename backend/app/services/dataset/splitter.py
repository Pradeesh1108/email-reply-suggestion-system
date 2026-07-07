"""
Deterministic train/holdout split for the email dataset.

Design note (accuracy §1): the holdout split is what makes aggregate "system
score" meaningful — holdout records are never seen by the retriever, so scores
reflect genuine generalization, not train-on-test overfitting.

The split is seeded and deterministic so results are reproducible.
"""

from __future__ import annotations

import random

from backend.app.domain.schemas import EmailRecord

SPLIT_SEED = 42
HOLDOUT_RATIO = 0.2


def split_dataset(
    records: list[EmailRecord],
    holdout_ratio: float = HOLDOUT_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[EmailRecord], list[EmailRecord]]:
    """
    Split records into (grounding_set, holdout_set).

    The grounding set is used to build the retrieval index.
    The holdout set is used only for evaluation — never seen by the retriever.

    Returns:
        (grounding_set, holdout_set)
    """
    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    holdout_count = max(1, int(len(records) * holdout_ratio))
    holdout_indices = set(indices[:holdout_count])

    grounding = [r for i, r in enumerate(records) if i not in holdout_indices]
    holdout = [r for i, r in enumerate(records) if i in holdout_indices]

    return grounding, holdout
