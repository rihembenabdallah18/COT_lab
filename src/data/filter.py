"""Stage 2: build the four training sets from GSM8K train + Ho et al.'s
Zero-shot-CoT teacher completions.

Outputs (one JSONL each, under data/processed/):

  set_a_nofilter.jsonl      all teacher CoTs (no filter)
  set_b_magister.jsonl      teacher CoTs whose final answer == gold
  set_c_calculator.jsonl    teacher CoTs whose final answer == gold AFTER
                            running each `A op B = C` through a calculator
  direct_ft.jsonl           one record per train example, with cot=""
                            (target collapses to ` #### {gold}`) — reproduces
                            Ho et al.'s "fine-tuning" reference (4.93%)

Common record schema:
  {sample_index, question, cot, gold_answer, teacher_predicted_answer,
   calculator_corrected_cot?, n_calc_edits?}

Also writes a Stage 2 run-card to outputs/runs/02_filter.json with sizes,
a Set-B/Set-C contingency table, and a few sample records.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.data.calculator import correct_equations
from src.data.parse_answer import parse_answer
from src.utils.runcard import finish, start

REPO_ROOT = Path(__file__).resolve().parents[2]
GSM8K_TRAIN = REPO_ROOT / "data" / "raw" / "gsm8k" / "train.jsonl"
HO_JSON = REPO_ROOT / "data" / "raw" / "ho_et_al_cots" / "gsm8k_zs_cot_text-davinci-002.json"
OUT_DIR = REPO_ROOT / "data" / "processed"
OUT_A = OUT_DIR / "set_a_nofilter.jsonl"
OUT_B = OUT_DIR / "set_b_magister.jsonl"
OUT_C = OUT_DIR / "set_c_calculator.jsonl"
OUT_DIRECT = OUT_DIR / "direct_ft.jsonl"

ANS_TOL = 1e-6


def _gold_str(parsed: float) -> str:
    """GSM8K answers are always integer-valued; render compactly."""
    return str(int(parsed)) if float(parsed).is_integer() else repr(parsed)


def _eq(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=ANS_TOL)


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "ans_tol": ANS_TOL,
        "calculator": "src.data.calculator.correct_equations",
        "tolerance_for_calculator_rewrite": "max(1e-6, 0.01 * max(|actual|, 1.0))",
    }
    card = start("02", "filter", config_snapshot)

    train = [json.loads(l) for l in GSM8K_TRAIN.open()]
    teacher_blob = json.loads(HO_JSON.read_text())
    teacher = teacher_blob["data"]

    # Counters
    n_total = 0
    n_set_a = 0
    n_set_b = 0
    n_set_c = 0
    n_direct = 0
    n_skipped_no_teacher = 0
    n_skipped_unparseable_gold = 0
    n_skipped_unparseable_teacher = 0

    # Set B / Set C contingency
    in_b_only = 0
    in_c_only = 0
    in_both = 0
    in_neither = 0
    n_calc_edited = 0

    samples_a: list[dict] = []
    samples_b: list[dict] = []
    samples_c: list[dict] = []
    samples_direct: list[dict] = []

    with OUT_A.open("w") as fa, OUT_B.open("w") as fb, \
         OUT_C.open("w") as fc, OUT_DIRECT.open("w") as fd:

        for i, ex in enumerate(train):
            n_total += 1

            gold = parse_answer(ex["answer"])
            if gold is None:
                n_skipped_unparseable_gold += 1
                continue

            # Direct FT — one row per train example, no teacher needed.
            direct_record = {
                "sample_index": i,
                "question": ex["question"],
                "cot": "",
                "gold_answer": _gold_str(gold),
                "teacher_predicted_answer": None,
            }
            fd.write(json.dumps(direct_record) + "\n")
            n_direct += 1
            if len(samples_direct) < 2:
                samples_direct.append(direct_record)

            recs = teacher.get(str(i))
            if not recs:
                n_skipped_no_teacher += 1
                continue
            t = recs[0]    # zs_cot has one completion per sample_index

            cot = (t.get("reasoning_completion") or "").strip()
            teacher_pred = parse_answer(t.get("completion"))

            base_record = {
                "sample_index": i,
                "question": ex["question"],
                "cot": cot,
                "gold_answer": _gold_str(gold),
                "teacher_predicted_answer": (t.get("completion") or "").strip(),
            }

            # Set A — always included (provided we have a teacher CoT).
            fa.write(json.dumps(base_record) + "\n")
            n_set_a += 1
            if len(samples_a) < 2:
                samples_a.append(base_record)

            # Set B — Magister's filter on the raw teacher prediction.
            in_b = teacher_pred is not None and _eq(teacher_pred, gold)
            if in_b:
                fb.write(json.dumps(base_record) + "\n")
                n_set_b += 1
                if len(samples_b) < 2:
                    samples_b.append(base_record)
            elif teacher_pred is None:
                n_skipped_unparseable_teacher += 1

            # Set C — calculator-corrected filter. We rewrite the CoT, re-parse
            # the final answer, and accept iff it matches gold.
            corrected_cot, edits = correct_equations(cot)
            n_calc_edits = len(edits)
            if n_calc_edits > 0:
                n_calc_edited += 1
            corrected_pred = parse_answer(corrected_cot)
            in_c = corrected_pred is not None and _eq(corrected_pred, gold)
            if in_c:
                rec_c = {
                    **base_record,
                    "cot": corrected_cot,             # train students on the corrected version
                    "calculator_corrected_cot": True,
                    "n_calc_edits": n_calc_edits,
                }
                fc.write(json.dumps(rec_c) + "\n")
                n_set_c += 1
                if len(samples_c) < 2:
                    samples_c.append(rec_c)

            # Contingency
            if in_b and in_c:
                in_both += 1
            elif in_b and not in_c:
                in_b_only += 1
            elif in_c and not in_b:
                in_c_only += 1
            else:
                in_neither += 1

    # ---- Console report -------------------------------------------------
    print(f"GSM8K train rows: {n_total}")
    print(f"  Direct FT      : {n_direct} -> {OUT_DIRECT}")
    print(f"  Set A (no flt) : {n_set_a} -> {OUT_A}")
    print(f"  Set B (Magist) : {n_set_b} -> {OUT_B}")
    print(f"  Set C (calc.)  : {n_set_c} -> {OUT_C}")
    print(f"  skipped: no_teacher={n_skipped_no_teacher} "
          f"unparseable_gold={n_skipped_unparseable_gold} "
          f"unparseable_teacher_pred={n_skipped_unparseable_teacher}")
    print(f"  Set B keep rate (of A): {n_set_b / max(n_set_a, 1):.1%}")
    print(f"  Set C keep rate (of A): {n_set_c / max(n_set_a, 1):.1%}")
    print(f"  CoTs the calculator edited: {n_calc_edited} / {n_set_a}")

    print("\n--- Set B / Set C contingency ---")
    print(f"  in both    : {in_both}")
    print(f"  Set C only : {in_c_only}   (chains rescued by calculator)")
    print(f"  Set B only : {in_b_only}   (calc broke a previously-correct answer)")
    print(f"  in neither : {in_neither}")

    print("\n--- 2 example records, Set A ---")
    for rec in samples_a:
        print(_short(rec))
    print("\n--- 2 example records, Set B ---")
    for rec in samples_b:
        print(_short(rec))
    print("\n--- 2 example records, Set C ---")
    for rec in samples_c:
        print(_short(rec))

    # ---- Run-card -------------------------------------------------------
    finish(
        card,
        metrics={
            "gsm8k_train_rows": n_total,
            "set_a_size": n_set_a,
            "set_b_size": n_set_b,
            "set_c_size": n_set_c,
            "direct_ft_size": n_direct,
            "set_b_keep_rate": round(n_set_b / max(n_set_a, 1), 4),
            "set_c_keep_rate": round(n_set_c / max(n_set_a, 1), 4),
            "n_calculator_edited": n_calc_edited,
            "skipped_no_teacher": n_skipped_no_teacher,
            "skipped_unparseable_gold": n_skipped_unparseable_gold,
            "skipped_unparseable_teacher_pred": n_skipped_unparseable_teacher,
            "contingency_b_and_c": in_both,
            "contingency_c_only": in_c_only,
            "contingency_b_only": in_b_only,
            "contingency_neither": in_neither,
        },
        inputs=[str(GSM8K_TRAIN.relative_to(REPO_ROOT)),
                str(HO_JSON.relative_to(REPO_ROOT))],
        outputs=[str(p.relative_to(REPO_ROOT)) for p in (OUT_A, OUT_B, OUT_C, OUT_DIRECT)],
        samples=[{"set": "A", **samples_a[0]} if samples_a else {},
                 {"set": "B", **samples_b[0]} if samples_b else {},
                 {"set": "C", **samples_c[0]} if samples_c else {},
                 {"set": "direct", **samples_direct[0]} if samples_direct else {}],
        notes="Stage 2 v2: emits Set A, Set B, Set C (calculator), and Direct FT.",
    )


def _short(rec: dict) -> str:
    """Pretty-print a record with long strings clipped to 120 chars."""
    return json.dumps({k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                       for k, v in rec.items()}, indent=2)


if __name__ == "__main__":
    build()
