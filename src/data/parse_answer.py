"""Parse the final numeric answer from a chain-of-thought string.

Used everywhere: gold answers (GSM8K format `#### <n>`), teacher final
completions (`" 72."`, `" $9.96."`), and student outputs (which we train to
emit `{cot} #### {gold_answer}`).

Returns a float, or None if no number is found. Caller decides equality
semantics — for GSM8K we use `abs(a - b) < 1e-6`.
"""
from __future__ import annotations

import re
from typing import Optional

# Number = optional sign, then either thousands-separated (1,234) or plain (1234),
# optionally followed by a decimal part. Order matters: the comma-grouped
# alternative must come first so 1,234 is matched as one number, not as 1 and 234.
_NUM_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_HASH_RE = re.compile(r"####\s*(" + _NUM_RE.pattern + ")")


def parse_answer(text: Optional[str]) -> Optional[float]:
    """Return the final numeric answer in `text`, or None if absent.

    Priority:
      1. The number after the last `####` marker (GSM8K's gold/target format).
      2. The last number anywhere in the string (free-text fallback).
    """
    if text is None:
        return None
    # Take the LAST #### occurrence in case the model emits more than one.
    hash_matches = list(_HASH_RE.finditer(text))
    if hash_matches:
        return _to_float(hash_matches[-1].group(1))
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    return _to_float(matches[-1])


def parse_answer_strict(text: Optional[str]) -> Optional[float]:
    """Return the answer only if the text contains a `#### N` marker.

    Unlike `parse_answer`, this never falls back to the last number in the
    string. Use this for a fair comparison against the baseline, which never
    emits `####` and would otherwise score via the fallback.
    """
    if text is None:
        return None
    hash_matches = list(_HASH_RE.finditer(text))
    if hash_matches:
        return _to_float(hash_matches[-1].group(1))
    return None


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))
