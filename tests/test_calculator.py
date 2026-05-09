"""Unit tests for the equation rewriter (Set C construction + acc-with-calc).

The rewriter must:
  - leave correct equations alone (including rounded approximations);
  - replace clearly wrong arithmetic with the true value, preserving the
    surrounding text and number formatting;
  - tolerate decimals, negatives, and division;
  - handle chains with no equations and chains with multiple equations.
"""
import math

import pytest

from src.data.calculator import correct_equations
from src.data.parse_answer import parse_answer


# ----- correctness preserved -----------------------------------------------

def test_correct_equation_unchanged():
    text = "She has 48 + 24 = 72 clips."
    out, edits = correct_equations(text)
    assert out == text
    assert edits == []


def test_rounded_decimal_within_tolerance_unchanged():
    # 50 / 60 = 0.8333..., teacher rounds to 0.83 — within 1% so kept.
    text = "She worked 50 / 60 = 0.83 hours."
    out, edits = correct_equations(text)
    assert out == text
    assert edits == []


def test_no_equations_unchanged():
    text = "Betty needs more money but the chain has no equations."
    out, edits = correct_equations(text)
    assert out == text
    assert edits == []


# ----- single-error rewrite -------------------------------------------------

def test_clear_arithmetic_error_replaced():
    # Teacher claims 50 + 10 = 70 (wrong; should be 60). Surrounding correct
    # equation (100 - 45 = 55) is left alone.
    text = "100 - 45 = 55. Also 50 + 10 = 70."
    out, edits = correct_equations(text)
    assert "100 - 45 = 55" in out
    assert "50 + 10 = 60" in out
    assert len(edits) == 1
    assert math.isclose(edits[0].actual, 60.0)
    assert math.isclose(edits[0].claimed, 70.0)


# ----- multi-equation chains ------------------------------------------------

def test_multiple_errors_all_fixed():
    text = "First 2 + 2 = 5. Then 3 * 3 = 10."
    out, edits = correct_equations(text)
    assert "2 + 2 = 4" in out
    assert "3 * 3 = 9" in out
    assert len(edits) == 2


# ----- edge cases ----------------------------------------------------------

def test_division_by_zero_left_alone():
    text = "Then 5 / 0 = 0 because the chain divides by zero."
    out, edits = correct_equations(text)
    assert out == text
    assert edits == []


def test_negative_result_replaced():
    # 3 - 10 = -7, teacher claims 7.
    text = "Net change: 3 - 10 = 7."
    out, edits = correct_equations(text)
    assert "3 - 10 = -7" in out
    assert len(edits) == 1


def test_decimal_arithmetic_replaced():
    # 1.5 * 4 = 6.0, teacher claims 5.0.
    text = "Total cost: 1.5 * 4 = 5.0."
    out, edits = correct_equations(text)
    assert "1.5 * 4 = 6" in out
    assert len(edits) == 1


# ----- integration with parse_answer ---------------------------------------

def test_parse_answer_after_rewrite_picks_corrected_value_when_chain_ends_in_equation():
    # Teacher miscomputes 12 * 52 = 600 (true value 624). Chain ends with the
    # wrong equation; after rewrite, parse_answer picks up 624.
    text = "Total pages = 12 * 52 = 600"
    out, edits = correct_equations(text)
    assert len(edits) == 1
    assert parse_answer(out) == 624.0


def test_parse_answer_unchanged_when_final_number_is_prose():
    # If the wrong number lives outside an equation (in trailing prose),
    # the calculator can't help — documented limitation.
    text = "He writes 12 * 52 = 600 pages a year. So the answer is 600."
    out, _ = correct_equations(text)
    # The equation rewrite happens, but parse_answer takes the LAST number,
    # which is the prose "600" — that's the documented limitation.
    assert parse_answer(out) == 600.0
