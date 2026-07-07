"""
Unit tests for the TF-IDF retriever.

Runs fully offline — no network access needed.
"""

import pytest

from backend.app.domain.schemas import EmailRecord, EmailMetadata
from backend.app.services.retrieval.tfidf_retriever import TfidfRetriever


def _make_record(id: str, email: str, reply: str = "Thanks", category: str = "customer_support") -> EmailRecord:
    return EmailRecord(
        id=id,
        category=category,
        incoming_email=email,
        sent_reply=reply,
        metadata=EmailMetadata(tone="formal", sender_role="customer", urgency="medium"),
    )


@pytest.fixture
def retriever():
    r = TfidfRetriever()
    r.fit([
        _make_record("r1", "I need help with my order #1234 that hasn't shipped yet"),
        _make_record("r2", "What are your API rate limits for the enterprise plan?", category="sales_inquiry"),
        _make_record("r3", "Can we reschedule our meeting to Thursday?", category="scheduling"),
        _make_record("r4", "I want a refund for the defective product I received"),
        _make_record("r5", "How do I reset my password? The reset email isn't arriving"),
    ])
    return r


class TestTfidfRetriever:
    def test_fit_and_top_k_returns_results(self, retriever):
        result = retriever.top_k("My order hasn't shipped", k=3)
        assert len(result.examples) == 3
        # The order-related query should match the order-related record best
        assert result.examples[0].record_id == "r1"

    def test_top_k_respects_k(self, retriever):
        result = retriever.top_k("order", k=2)
        assert len(result.examples) == 2

    def test_similarity_scores_are_sorted(self, retriever):
        result = retriever.top_k("order shipping delay", k=5)
        scores = [ex.similarity_score for ex in result.examples]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_scores_are_in_valid_range(self, retriever):
        result = retriever.top_k("testing query", k=5)
        for ex in result.examples:
            assert 0.0 <= ex.similarity_score <= 1.0

    def test_low_confidence_flag(self):
        """Queries with zero overlap should flag low confidence."""
        r = TfidfRetriever()
        r.fit([_make_record("r1", "completely unrelated topic about quantum physics")])
        result = r.top_k("xylophone marketplace zebra", k=1)
        # With nonsense query, score should be very low
        assert result.examples[0].similarity_score < 0.15 or result.low_confidence

    def test_not_fitted_raises(self):
        r = TfidfRetriever()
        with pytest.raises(RuntimeError, match="not fitted"):
            r.top_k("test query")

    def test_category_preserved(self, retriever):
        result = retriever.top_k("API rate limits", k=1)
        assert result.examples[0].category == "sales_inquiry"
