"""
Groq-based reply generator.

Design note (cost-control §6): exactly ONE LLM call per suggestion. The prompt
is built from the system instructions + few-shot retrieved examples + the new
email. Responses are cached by hash of (prompt_version, email, example_ids).
"""

from __future__ import annotations

import json

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.domain.schemas import GeneratedReply, RetrievedExample
from backend.app.infra.cache import ResponseCache
from backend.app.infra.groq_client import GroqClient
from backend.app.services.generation.base import Generator
from backend.app.services.generation.prompt_templates import (
    GENERATION_PROMPT_VERSION,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_TEMPLATE,
    format_examples_block,
)

logger = get_logger(__name__)


class GroqGenerator(Generator):
    """Concrete generator using the Groq API."""

    def __init__(
        self,
        groq_client: GroqClient,
        cache: ResponseCache | None = None,
    ) -> None:
        self._client = groq_client
        self._cache = cache

    def generate(
        self,
        incoming_email: str,
        examples: list[RetrievedExample],
    ) -> GeneratedReply:
        """
        Generate a reply via one Groq chat-completion call.

        Cache key = hash(prompt_version, incoming_email, retrieved_example_ids).
        """
        settings = get_settings()
        example_ids = [ex.record_id for ex in examples]

        # Build few-shot examples block
        examples_block = format_examples_block(
            [
                {"incoming_email": ex.incoming_email, "sent_reply": ex.sent_reply}
                for ex in examples
            ]
        )

        user_message = GENERATION_USER_TEMPLATE.format(
            examples_block=examples_block,
            incoming_email=incoming_email,
        )

        # Cache key from prompt version + email + example IDs
        cache_key = None
        if self._cache:
            cache_key = ResponseCache.make_key(
                prompt_version=GENERATION_PROMPT_VERSION,
                incoming_email=incoming_email,
                retrieved_ids=json.dumps(example_ids),
            )

        text = self._client.chat_completion(
            model=settings.groq_model_generator,
            messages=[
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=1024,
            cache_key=cache_key,
        )

        low_confidence = any(
            getattr(ex, "similarity_score", 1.0) < settings.retrieval_confidence_threshold
            for ex in examples
        )

        return GeneratedReply(
            text=text.strip(),
            retrieved_example_ids=example_ids,
            low_retrieval_confidence=low_confidence,
            prompt_version=GENERATION_PROMPT_VERSION,
            model_used=settings.groq_model_generator,
        )
