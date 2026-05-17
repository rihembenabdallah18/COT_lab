"""Intra-step coherence (simplified RCU).

For each step, compute P(entailment | premise=step, hypothesis=step).
Chain score = min over steps.

Limitation: with premise == hypothesis, NLI scores are near-trivially high
for any grammatical sentence; this is a baseline-only approximation that
matches the ReCEval paper's simplified RCU. A proper check would split each
step into evidence and claim. Flagged in the write-up.
"""
from __future__ import annotations

from . import _nli


def score_chain(steps: list[str], batch_size: int = 16) -> float:
    if not steps:
        return float("nan")
    pairs = [(s, s) for s in steps]
    probs = _nli.batch_probs(pairs, label="entailment", batch_size=batch_size)
    return min(probs)


def score_steps(steps: list[str], batch_size: int = 16) -> list[float]:
    if not steps:
        return []
    pairs = [(s, s) for s in steps]
    return _nli.batch_probs(pairs, label="entailment", batch_size=batch_size)