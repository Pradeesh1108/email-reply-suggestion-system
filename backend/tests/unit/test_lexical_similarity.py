"""
Unit tests for lexical similarity.

Runs fully offline — no network access needed.
"""

import pytest

from backend.app.services.evaluation.lexical_similarity import compute_lexical_similarity


class TestLexicalSimilarity:
    def test_identical_texts_high_similarity(self):
        text = "Thank you for your email. I'll process this right away."
        sim = compute_lexical_similarity(text, text)
        assert sim > 0.99

    def test_similar_texts_moderate_similarity(self):
        candidate = "Thank you for reaching out about your order. I'll check on the status."
        reference = "Thanks for contacting us about your order. Let me look into the status."
        sim = compute_lexical_similarity(candidate, reference)
        assert 0.2 < sim < 1.0

    def test_unrelated_texts_low_similarity(self):
        candidate = "The weather forecast shows rain tomorrow morning."
        reference = "Your order has been shipped via express delivery."
        sim = compute_lexical_similarity(candidate, reference)
        assert sim < 0.3

    def test_empty_candidate_returns_zero(self):
        sim = compute_lexical_similarity("", "Some reference text")
        assert sim == 0.0

    def test_empty_reference_returns_zero(self):
        sim = compute_lexical_similarity("Some candidate text", "")
        assert sim == 0.0

    def test_both_empty_returns_zero(self):
        sim = compute_lexical_similarity("", "")
        assert sim == 0.0

    def test_score_in_valid_range(self):
        sim = compute_lexical_similarity(
            "Hello world, this is a test email reply.",
            "Greetings, this is a completely different response."
        )
        assert 0.0 <= sim <= 1.0
