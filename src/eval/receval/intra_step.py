"""Intra-step coherence (simplified RCU).

For each step, compute P(entailment | premise=step, hypothesis=step).
Chain score = min over steps.

LIMITATION: with premise == hypothesis the NLI model returns near-trivially
high entailment scores (~1.0) for any grammatical sentence. Meaningful
variation across conditions is therefore not expected from this metric in
isolation. A proper intra-step check would split each step into supporting
evidence and claimed conclusion, then verify P(entailment | evidence, claim).
This simplified formulation is retained to match the spec and documents the
baseline limitation; it should be flagged as such in the paper.
"""
from __future__ import annotations

from . import _nli


def score_chain(steps: list[str], batch_size: int = 16) -> float:
    """Return min P(entailment) over steps (premise = hypothesis = step)."""
    if not steps:
        return float("nan")
    pairs = [(s, s) for s in steps]
    probs = _nli.batch_probs(pairs, label="entailment", batch_size=batch_size)
    return min(probs)


def score_steps(steps: list[str], batch_size: int = 16) -> list[float]:
    """Return per-step P(entailment) for detailed inspection."""
    if not steps:
        return []
    pairs = [(s, s) for s in steps]
    return _nli.batch_probs(pairs, label="entailment", batch_size=batch_size)
