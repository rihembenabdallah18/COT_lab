"""Parse the final numeric answer from a chain-of-thought string.

Used everywhere: gold answers (GSM8K format `#### <n>`), teacher final
completions (`" 72."`, `" $9.96."`), and student outputs (trained to emit
`{cot} #### {gold_answer}`).

Returns a float, or None if no number is found. Caller decides equality
semantics; for GSM8K we use `abs(a - b) < 1e-6`.
"""
from __future__ import annotations

import re
from typing import Optional

_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_HASH_RE = re.compile(r"####\s*(" + _NUM_RE.pattern + ")")


def parse_answer(text: Optional[str]) -> Optional[float]:
    """Return the final numeric answer in `text`, or None if absent.

    Priority:
      1. Number after the last `####` marker (GSM8K's gold/target format).
      2. Last number anywhere in the string (free-text fallback).
    """
    if text is None:
        return None
    hash_matches = list(_HASH_RE.finditer(text))
    if hash_matches:
        return _to_float(hash_matches[-1].group(1))
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    return _to_float(matches[-1])


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))
