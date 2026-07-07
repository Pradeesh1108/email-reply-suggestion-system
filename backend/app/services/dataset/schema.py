"""
JSONL record schema validation for the email dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.domain.schemas import EmailRecord


def validate_record(raw: dict) -> EmailRecord:
    """Parse and validate a single JSONL record into an EmailRecord."""
    return EmailRecord.model_validate(raw)


def validate_jsonl_file(path: Path) -> list[EmailRecord]:
    """Load and validate every record in a JSONL file."""
    records: list[EmailRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                records.append(validate_record(raw))
            except Exception as e:
                raise ValueError(f"Line {line_num} in {path}: {e}") from e
    return records
