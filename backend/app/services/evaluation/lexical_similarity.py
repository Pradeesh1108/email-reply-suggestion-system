"""
TF-IDF cosine similarity between candidate and reference replies.

Design note (accuracy §1): this is a SECONDARY sanity signal, reported as a
clearly-labeled separate field — never folded into the composite score.  Two
very different replies can both be excellent, so low lexical similarity does
NOT mean low quality.  This is the most important accuracy-definition decision
in the project and is documented in the README's weighting rationale section.

Cost: zero API tokens (pure scikit-learn computation).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_lexical_similarity(
    candidate_reply: str,
    reference_reply: str,
) -> float:
    """
    Compute TF-IDF cosine similarity between candidate and reference replies.

    Returns:
        Similarity score in [0, 1].  This is NOT a quality score — it's a
        sanity check for how lexically similar the candidate is to the
        reference.  Interpret with caution (see module docstring).
    """
    if not candidate_reply.strip() or not reference_reply.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([candidate_reply, reference_reply])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim), 4)
    except ValueError:
        # Edge case: both texts reduce to nothing after stop-word removal
        return 0.0
