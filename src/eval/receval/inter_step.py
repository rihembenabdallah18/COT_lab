"""Inter-step coherence.

For step i, build prior context = [question, step_0, …, step_{i-1}].
For each prior element r, compute P(contradiction | premise=r, hypothesis=step_i).
Step score = 1 − max_r P(contradiction).
Chain score = min over steps.
"""
from __future__ import annotations

from . import _nli


def score_chain(question: str, steps: list[str], batch_size: int = 16) -> float:
    """Return min inter-step coherence score over all steps."""
    if not steps:
        return float("nan")
    step_scores = score_steps(question, steps, batch_size=batch_size)
    return min(step_scores)


def score_steps(question: str, steps: list[str], batch_size: int = 16) -> list[float]:
    """Return per-step inter-step coherence scores for detailed inspection."""
    if not steps:
        return []
    all_pairs: list[tuple[str, str]] = []
    boundaries: list[int] = []
    for i, step in enumerate(steps):
        prior = [question] + steps[:i]
        for r in prior:
            all_pairs.append((r, step))
        boundaries.append(len(all_pairs))

    all_contra = _nli.batch_probs(all_pairs, label="contradiction", batch_size=batch_size)

    step_scores: list[float] = []
    prev = 0
    for end in boundaries:
        chunk = all_contra[prev:end]
        step_scores.append(1.0 - max(chunk))
        prev = end
    return step_scores
