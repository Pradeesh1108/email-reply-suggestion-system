"""
Abstract base class for retrievers.

Design note (modularity §3): every services/* subpackage exposes an ABC so the
concrete implementation (TF-IDF today, embedding-based later) can be swapped
without touching calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.domain.schemas import EmailRecord, RetrievalResult


class Retriever(ABC):
    """Interface for retrieval of similar past email/reply pairs."""

    @abstractmethod
    def fit(self, corpus: list[EmailRecord]) -> None:
        """
        Build the retrieval index from the grounding corpus.

        This should be called once at startup (or on dataset change), never
        per request (cost-control §6).
        """
        ...

    @abstractmethod
    def top_k(self, query: str, k: int = 3) -> RetrievalResult:
        """
        Retrieve the k most similar past email/reply pairs.

        Args:
            query: The incoming email text to match against.
            k: Number of examples to return.

        Returns:
            RetrievalResult with ranked examples and a low_confidence flag.
        """
        ...
