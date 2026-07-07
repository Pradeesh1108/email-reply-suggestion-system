"""
Unit tests for rule-based checks.

Runs fully offline — no network access needed.
"""

import pytest

from backend.app.services.evaluation.rule_checks import (
    check_placeholder_leak,
    check_empty_or_degenerate,
    check_naive_hallucination,
    run_all_checks,
)


class TestPlaceholderLeak:
    def test_clean_reply_passes(self):
        result = check_placeholder_leak("Thank you for reaching out. I'll look into this.")
        assert result.passed is True
        assert result.hard_failure is False

    def test_bracket_placeholder_fails(self):
        result = check_placeholder_leak("Dear [Your Name], thanks for contacting us.")
        assert result.passed is False
        assert result.hard_failure is True
        assert any("[Your Name]" in item for item in result.flagged_items)

    def test_mustache_placeholder_fails(self):
        result = check_placeholder_leak("Your refund of {{refund_amount}} has been processed.")
        assert result.passed is False
        assert result.hard_failure is True

    def test_angle_bracket_placeholder_fails(self):
        result = check_placeholder_leak("Please contact <PLACEHOLDER> for help.")
        assert result.passed is False
        assert result.hard_failure is True

    def test_multiple_placeholders(self):
        reply = "Dear [Customer Name], your {{product}} order is ready. Contact [Your Name]."
        result = check_placeholder_leak(reply)
        assert len(result.flagged_items) >= 3


class TestEmptyOrDegenerate:
    def test_normal_reply_passes(self):
        result = check_empty_or_degenerate(
            "Thank you for your email. I'll process your request right away.",
            "I have a question about my order."
        )
        assert result.passed is True

    def test_empty_reply_is_hard_failure(self):
        result = check_empty_or_degenerate("", "I have a question.")
        assert result.passed is False
        assert result.hard_failure is True

    def test_near_empty_reply_is_hard_failure(self):
        result = check_empty_or_degenerate("Ok", "I have a question about my order.")
        assert result.passed is False
        assert result.hard_failure is True

    def test_ridiculously_long_reply(self):
        short_email = "Help please."
        long_reply = "word " * 5000  # way too long
        result = check_empty_or_degenerate(long_reply, short_email)
        assert result.passed is False
        assert result.hard_failure is False  # not a hard failure, just a flag


class TestNaiveHallucination:
    def test_grounded_reply_passes(self):
        email = "I was charged $29 on June 15th for order #1234."
        reply = "I see the charge of $29 on your order #1234 from June 15th."
        result = check_naive_hallucination(reply, email)
        assert result.passed is True

    def test_hallucinated_amount_flagged(self):
        email = "I was charged $29 for my subscription."
        reply = "I've processed a refund of $150 to your account."
        result = check_naive_hallucination(reply, email)
        assert result.passed is False
        assert any("$150" in item for item in result.flagged_items)

    def test_reference_reply_is_checked(self):
        """Items in the reference reply should NOT be flagged."""
        email = "Please help with my order."
        reference = "Your refund of $89 has been processed."
        reply = "I've confirmed your refund of $89."
        result = check_naive_hallucination(reply, email, reference)
        # $89 is in the reference, so it should not be flagged
        assert not any("$89" in item for item in result.flagged_items)

    def test_common_phrases_not_flagged(self):
        email = "Hello, I need help."
        reply = "Thank you for reaching out.\n\nBest Regards,\nSupport Team"
        result = check_naive_hallucination(reply, email)
        # "Best Regards" and "Support Team" are common phrases, not hallucinations
        assert not any("Best Regards" in item for item in result.flagged_items)


class TestRunAllChecks:
    def test_clean_reply_all_pass(self):
        summary = run_all_checks(
            candidate_reply="Thank you for your email. I've processed your request.",
            incoming_email="I need help with my account.",
        )
        assert summary.any_hard_failure is False

    def test_placeholder_triggers_hard_failure(self):
        summary = run_all_checks(
            candidate_reply="Dear [Your Name], thank you for contacting {{company}}.",
            incoming_email="I need help.",
        )
        assert summary.any_hard_failure is True

    def test_empty_triggers_hard_failure(self):
        summary = run_all_checks(
            candidate_reply="",
            incoming_email="I need help.",
        )
        assert summary.any_hard_failure is True
