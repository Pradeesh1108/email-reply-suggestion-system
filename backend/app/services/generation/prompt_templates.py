"""
Versioned prompt templates for reply generation and LLM judging.

Design note (cost-control §6): the prompt version is part of the cache key,
so prompt changes automatically invalidate the cache without a manual flush.
"""

# ─── Generation prompt ────────────────────────────────────────────────────────

GENERATION_PROMPT_VERSION = "v1"

GENERATION_SYSTEM_PROMPT = """\
You are a professional email assistant. Your job is to draft a reply to the \
incoming email on behalf of the business.

Rules:
1. Match the tone and structure shown in the example replies below.
2. Address every distinct question or request in the incoming email.
3. Do NOT invent facts, commitments, prices, dates, or information that is \
   not present in or reasonably implied by the incoming email.
4. Include appropriate greeting and sign-off.
5. Be concise but complete — don't leave anything dangling.
6. Do NOT include template placeholders like [Your Name] or {{company}}.
"""

GENERATION_USER_TEMPLATE = """\
Here are examples of past emails and the replies that were sent:

{examples_block}

---

Now draft a reply to this new incoming email:

{incoming_email}

Reply:"""


def format_examples_block(examples: list[dict]) -> str:
    """Format retrieved examples into the few-shot block."""
    parts = []
    for i, ex in enumerate(examples, 1):
        parts.append(
            f"Example {i}:\n"
            f"Incoming email:\n{ex['incoming_email']}\n\n"
            f"Reply sent:\n{ex['sent_reply']}"
        )
    return "\n\n---\n\n".join(parts)


# ─── Judge prompt ─────────────────────────────────────────────────────────────

JUDGE_PROMPT_VERSION = "v1"

JUDGE_SYSTEM_PROMPT = """\
You are an expert email quality evaluator. You will be given an incoming email \
and a candidate reply. Score the reply on the following axes (1-5 scale each) \
and provide a short rationale for each score.

Axes:
1. relevance_coverage (1-5): Does the reply address every distinct ask/question \
   in the incoming email? 1 = misses most, 5 = covers everything.
2. groundedness (1-5): Does the reply avoid inventing commitments, facts, numbers, \
   or promises not present in or implied by the incoming email? 1 = fabricates heavily, \
   5 = fully grounded.
3. tone_fit (1-5): Does the reply match the appropriate register/formality implied \
   by the incoming email? 1 = wildly off-tone, 5 = perfect match.
4. completeness_actionability (1-5): Does the reply include proper sign-off, \
   next steps, and leave nothing dangling? 1 = incomplete/useless, 5 = fully actionable.

You MUST respond with ONLY valid JSON in this exact format, no other text:
{
  "relevance_coverage": {"score": <1-5>, "rationale": "<brief explanation>"},
  "groundedness": {"score": <1-5>, "rationale": "<brief explanation>"},
  "tone_fit": {"score": <1-5>, "rationale": "<brief explanation>"},
  "completeness_actionability": {"score": <1-5>, "rationale": "<brief explanation>"},
  "overall_rationale": "<1-2 sentence summary of overall quality>"
}
"""

JUDGE_USER_TEMPLATE = """\
Incoming email:
{incoming_email}

Candidate reply to evaluate:
{candidate_reply}

{hallucination_context}

Respond with ONLY the JSON evaluation:"""


def format_hallucination_context(flagged_items: list[str]) -> str:
    """Add hallucination flags as context for the judge."""
    if not flagged_items:
        return ""
    items = ", ".join(f'"{item}"' for item in flagged_items[:10])
    return (
        f"Note: An automated check flagged these items in the reply as "
        f"potentially not present in the incoming email: {items}. "
        f"Consider this when scoring groundedness, but use your own judgment."
    )
