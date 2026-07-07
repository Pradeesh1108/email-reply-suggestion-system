"""
SQLite-backed response cache keyed by content hash.

Design note (cost-control §6): repeated/duplicate requests (re-running holdout
eval, user re-submitting the same email) must not re-spend API tokens.  Cache
key = SHA-256 of (prompt_template_version, incoming_email_text, retrieved_example_ids).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """Simple key-value cache backed by a SQLite file."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def make_key(**components: str | list[str]) -> str:
        """
        Build a deterministic cache key from arbitrary string components.

        Usage:
            key = cache.make_key(
                prompt_version="v1",
                incoming_email="...",
                retrieved_ids=["cs_001", "si_003"],
            )
        """
        # Sort keys so order doesn't matter
        canonical = json.dumps(
            {k: v for k, v in sorted(components.items())},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Return cached value or None."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row:
            logger.debug("Cache hit: %s", key[:16])
            return row[0]
        return None

    def set(self, key: str, value: str) -> None:
        """Store a value in the cache (upsert)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, value),
            )
        logger.debug("Cache set: %s", key[:16])

    def clear(self) -> None:
        """Remove all cached entries."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM cache")
