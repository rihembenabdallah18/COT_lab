"""Equation rewriter used by Stage 2 to build Set C.

For each `A op B = C` substring in a teacher CoT, replace `C` with the
correct value when it differs from `lhs op rhs` by more than a small
tolerance. A claimed result is left alone when:

  abs(actual - claimed) <= max(1e-6, 0.01 * max(|actual|, 1.0))

This window is wide enough to leave 50/60 = 0.83 alone (rounded) but
narrow enough to catch genuine arithmetic mistakes like 6 * 52 = 312.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_EQ_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*"
    r"([+\-*/])\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*=\s*"
    r"([-+]?\d+(?:\.\d+)?)"
)

_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else None,
}


@dataclass
class Edit:
    span: tuple[int, int]
    original: str
    corrected: str
    claimed: float
    actual: float


def _format_number(x: float) -> str:
    if x == int(x):
        return str(int(x))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _is_close(claimed: float, actual: float) -> bool:
    threshold = max(1e-6, 0.01 * max(abs(actual), 1.0))
    return abs(actual - claimed) <= threshold


def correct_equations(text: str) -> tuple[str, list[Edit]]:
    """Rewrite each `A op B = C` in `text` with the correct value.

    Returns ``(rewritten_text, edits)``. ``edits`` is empty when the chain
    has no arithmetic errors (or no equations at all).
    """
    edits: list[Edit] = []

    def repl(m: re.Match) -> str:
        a = float(m.group(1))
        op = m.group(2)
        b = float(m.group(3))
        c = float(m.group(4))
        actual = _OPS[op](a, b)
        if actual is None:
            return m.group(0)
        if _is_close(c, actual):
            return m.group(0)
        new_c = _format_number(actual)
        new_full = f"{m.group(1)} {op} {m.group(3)} = {new_c}"
        edits.append(Edit(span=m.span(), original=m.group(0),
                          corrected=new_full, claimed=c, actual=actual))
        return new_full

    rewritten = _EQ_RE.sub(repl, text)
    return rewritten, edits
