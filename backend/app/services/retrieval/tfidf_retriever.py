"""
TF-IDF + cosine similarity retriever.

Design note (cost-control §1): at this dataset scale (tens to low hundreds of
examples), TF-IDF is perfectly adequate and honest.  It costs zero API tokens,
needs no GPU, and keeps the system CPU-only / installable in seconds.  The
trade-off is lexical-only matching (no semantic similarity) — documented in
the README's limitations section.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger
from backend.app.domain.schemas import EmailRecord, RetrievalResult, RetrievedExample
from backend.app.services.retrieval.base import Retriever

logger = get_logger(__name__)


class TfidfRetriever(Retriever):
    """Concrete retriever using TF-IDF vectorization and cosine similarity."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        self._corpus: list[EmailRecord] = []
        self._tfidf_matrix = None
        self._fitted = False

    def fit(self, corpus: list[EmailRecord]) -> None:
        """
        Build the TF-IDF index from incoming_email texts.

        Called once at startup (cost-control §6: index built once, not per request).
        """
        self._corpus = corpus
        texts = [r.incoming_email for r in corpus]
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        logger.info("TF-IDF retriever fitted on %d documents", len(corpus))

    def top_k(self, query: str, k: int = 3) -> RetrievalResult:
        """Retrieve k most similar past examples via cosine similarity."""
        if not self._fitted:
            raise RuntimeError("Retriever not fitted — call fit() first")

        settings = get_settings()
        k = min(k, len(self._corpus))

        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Rank by similarity (descending)
        top_indices = np.argsort(similarities)[::-1][:k]

        examples: list[RetrievedExample] = []
        for idx in top_indices:
            record = self._corpus[idx]
            examples.append(
                RetrievedExample(
                    record_id=record.id,
                    incoming_email=record.incoming_email,
                    sent_reply=record.sent_reply,
                    similarity_score=round(float(similarities[idx]), 4),
                    category=record.category,
                )
            )

        # Flag low confidence if best match is below threshold
        best_score = examples[0].similarity_score if examples else 0.0
        low_confidence = best_score < settings.retrieval_confidence_threshold

        if low_confidence:
            logger.warning(
                "Low retrieval confidence: best score %.4f < threshold %.4f",
                best_score,
                settings.retrieval_confidence_threshold,
            )

        return RetrievalResult(examples=examples, low_confidence=low_confidence)
