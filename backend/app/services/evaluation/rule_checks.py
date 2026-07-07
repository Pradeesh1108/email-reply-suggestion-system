"""
Rule-based checks — zero-cost heuristics that run before the LLM judge.

Design note (cost-control §1): these cost zero API tokens.  If a hard failure
is detected (placeholder leak or empty/degenerate reply), the aggregator can
short-circuit and skip the expensive judge call entirely.

Limitation (stated honestly per §1): the hallucination check is a regex
heuristic, NOT NER.  It catches obvious fabricated numbers/dates/names but
will miss more subtle hallucinations.  The LLM judge is the real safety net.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.app.domain.schemas import RuleCheckResult, RuleCheckSummary


def run_all_checks(
    candidate_reply: str,
    incoming_email: str,
    reference_reply: Optional[str] = None,
) -> RuleCheckSummary:
    """Run all rule-based checks and return a summary."""
    checks = [
        check_placeholder_leak(candidate_reply),
        check_empty_or_degenerate(candidate_reply, incoming_email),
        check_naive_hallucination(candidate_reply, incoming_email, reference_reply),
    ]

    any_hard = any(c.hard_failure for c in checks)
    total_flags = sum(len(c.flagged_items) for c in checks)

    return RuleCheckSummary(
        checks=checks,
        any_hard_failure=any_hard,
        total_flags=total_flags,
    )


def check_placeholder_leak(candidate_reply: str) -> RuleCheckResult:
    """
    Detect unresolved template placeholders in the reply.

    Patterns: [Your Name], [Customer Name], {{variable}}, <PLACEHOLDER>
    This is a hard failure — a reply with placeholders is obviously broken.
    """
    patterns = [
        (r'\[(?:Your|My|Customer|Client|Company)\s+\w+\]', "bracket placeholder"),
        (r'\{\{[^}]+\}\}', "mustache placeholder"),
        (r'<(?:PLACEHOLDER|INSERT|TODO|FILL)[^>]*>', "angle-bracket placeholder"),
    ]

    flagged: list[str] = []
    for pattern, label in patterns:
        matches = re.findall(pattern, candidate_reply, re.IGNORECASE)
        flagged.extend(matches)

    return RuleCheckResult(
        name="placeholder_leak",
        passed=len(flagged) == 0,
        hard_failure=len(flagged) > 0,  # hard failure — skip judge call
        detail=f"Found {len(flagged)} placeholder(s)" if flagged else "No placeholders detected",
        flagged_items=flagged,
    )


def check_empty_or_degenerate(
    candidate_reply: str,
    incoming_email: str,
) -> RuleCheckResult:
    """
    Check for empty, near-empty, or absurdly long replies.

    Thresholds:
    - Empty/near-empty: < 10 characters
    - Ridiculously long: > 10× the incoming email length
    """
    reply_len = len(candidate_reply.strip())
    email_len = max(len(incoming_email.strip()), 1)

    flagged: list[str] = []

    if reply_len < 10:
        flagged.append(f"Reply too short ({reply_len} chars)")

    if reply_len > email_len * 10:
        flagged.append(
            f"Reply suspiciously long ({reply_len} chars vs {email_len} char email, "
            f"ratio {reply_len / email_len:.1f}×)"
        )

    return RuleCheckResult(
        name="empty_or_degenerate",
        passed=len(flagged) == 0,
        hard_failure=reply_len < 10,  # hard failure only for empty/near-empty
        detail=flagged[0] if flagged else "Length is reasonable",
        flagged_items=flagged,
    )


def check_naive_hallucination(
    candidate_reply: str,
    incoming_email: str,
    reference_reply: Optional[str] = None,
) -> RuleCheckResult:
    """
    Flag items in the reply that aren't in the source email (or reference).

    Extracts:
    - Numbers / currency amounts (e.g., $500, 42, 3.5%)
    - Dates (e.g., June 15, 2024-01-01, 7/15)
    - Capitalized multi-word phrases (potential proper nouns like "John Smith")

    This is a HEURISTIC, not NER — it will produce false positives (e.g., generic
    phrases like "Best regards" get flagged as capitalized phrases). The LLM judge
    should reason about these flags, not just repeat them. See README limitations.
    """
    # Build the "known text" from the incoming email + reference (if present)
    known_text = incoming_email.lower()
    if reference_reply:
        known_text += " " + reference_reply.lower()

    reply_lower = candidate_reply.lower()
    flagged: list[str] = []

    # ── Extract numbers / currency ────────────────────────────────────────
    # Match patterns like $500, €200, 42, 3.5%, etc.
    number_pattern = r'(?:[\$€£¥][\d,]+(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%)?)'
    reply_numbers = set(re.findall(number_pattern, candidate_reply))
    for num in reply_numbers:
        if num.lower() not in known_text:
            flagged.append(f"number/currency: {num}")

    # ── Extract dates ─────────────────────────────────────────────────────
    date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s*\d{4})?\b',
        r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
    ]
    for pattern in date_patterns:
        reply_dates = re.findall(pattern, candidate_reply, re.IGNORECASE)
        for date_str in reply_dates:
            if date_str.lower() not in known_text:
                flagged.append(f"date: {date_str}")

    # ── Extract capitalized multi-word phrases (potential proper nouns) ───
    # Match 2+ consecutive capitalized words (not at sentence start)
    proper_noun_pattern = r'(?<=[.!?\n]\s)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    # Also match mid-sentence capitalized phrases
    mid_sentence_pattern = r'(?<=\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    proper_nouns = set(re.findall(mid_sentence_pattern, candidate_reply))

    # Filter out common non-hallucination phrases
    common_phrases = {
        "Best Regards", "Kind Regards", "Warm Regards", "Best Wishes",
        "Thank You", "Dear Sir", "Dear Madam", "Customer Support",
        "Support Team", "Sales Team", "Billing Team", "Billing Department",
        "Customer Experience", "App Store",
    }
    for phrase in proper_nouns:
        if phrase in common_phrases:
            continue
        if phrase.lower() not in known_text:
            flagged.append(f"proper noun: {phrase}")

    return RuleCheckResult(
        name="naive_hallucination_flags",
        passed=len(flagged) == 0,
        hard_failure=False,  # soft flag — let the judge reason about it
        detail=f"Flagged {len(flagged)} item(s) not found in source" if flagged else "No hallucination flags",
        flagged_items=flagged,
    )
