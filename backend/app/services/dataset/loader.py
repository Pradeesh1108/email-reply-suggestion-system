"""
Dataset loader — loads email_reply_dataset.jsonl and calibration_set.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.domain.schemas import CalibrationItem, EmailRecord


def load_dataset(path: Path) -> list[EmailRecord]:
    """Load the full email/reply dataset from JSONL."""
    records: list[EmailRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(EmailRecord.model_validate(json.loads(line)))
    return records


def load_calibration_set(path: Path) -> list[CalibrationItem]:
    """Load the calibration set from JSONL."""
    items: list[CalibrationItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(CalibrationItem.model_validate(json.loads(line)))
    return items
