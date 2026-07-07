"""
Abstract base class for reply generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.app.domain.schemas import GeneratedReply, RetrievedExample


class Generator(ABC):
    """Interface for LLM-based reply generation."""

    @abstractmethod
    def generate(
        self,
        incoming_email: str,
        examples: list[RetrievedExample],
    ) -> GeneratedReply:
        """
        Generate a suggested reply for the incoming email.

        Args:
            incoming_email: The new email to reply to.
            examples: Retrieved past (email, reply) pairs for few-shot grounding.

        Returns:
            GeneratedReply with text, provenance, and metadata.
        """
        ...
