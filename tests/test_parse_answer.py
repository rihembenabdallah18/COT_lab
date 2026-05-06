"""Unit tests for parse_answer. Covers GSM8K gold format, teacher completions,
student outputs, and edge cases."""
import math

import pytest

from src.data.parse_answer import parse_answer


@pytest.mark.parametrize("text, expected", [
    # --- GSM8K gold format ---
    ("#### 72", 72.0),
    (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
        "#### 72",
        72.0,
    ),
    ("#### 0", 0.0),  # zero must not be confused with None
    ("#### 1,234,567", 1234567.0),
    ("#### -3", -3.0),
    # --- Teacher final completion (Ho et al.) ---
    (" 72.", 72.0),
    (" $9.96.", 9.96),
    (" 312 pages.", 312.0),
    # --- Free-text fallback: take the LAST number ---
    ("Walks 3 miles, runs 4 miles. Total: 7.", 7.0),
    ("price was $5.50 then $6.75", 6.75),
    ("The answer is -2", -2.0),
    # --- Student output style (after training on `{cot} #### {gold}`) ---
    ("She has 12 apples then buys 8 more. #### 20", 20.0),
    # --- Edge cases ---
    ("", None),
    ("no numbers here at all", None),
    (None, None),
    # Decimal with no leading digit before fallback — we don't try to support `.5`,
    # but a bare integer is fine.
    ("3.14", 3.14),
    # Multiple #### — keep the LAST one (a model that re-asserts wins).
    ("#### 5\nactually let me recompute\n#### 8", 8.0),
])
def test_parse_answer(text, expected):
    got = parse_answer(text)
    if expected is None:
        assert got is None
    else:
        assert got is not None and math.isclose(got, expected)
