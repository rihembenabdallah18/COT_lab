"""Stage 2: build Set A (no filter) and Set B (Magister answer-correctness filter)
from GSM8K train + Ho et al. teacher Zero-shot-CoT completions.

Output records: {"sample_index", "question", "cot", "gold_answer",
                 "teacher_predicted_answer"}.
- `cot` = teacher's `reasoning_completion` (the step-by-step reasoning).
- `gold_answer` is a string (e.g. "72") matching GSM8K's `#### N` convention.
- `teacher_predicted_answer` is the teacher's final-answer extraction string,
  retained for diagnostics (Set B keeps only records where the parsed value
  matches gold).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.data.parse_answer import parse_answer

REPO_ROOT = Path(__file__).resolve().parents[2]
GSM8K_TRAIN = REPO_ROOT / "data" / "raw" / "gsm8k" / "train.jsonl"
HO_JSON = REPO_ROOT / "data" / "raw" / "ho_et_al_cots" / "gsm8k_zs_cot_text-davinci-002.json"
OUT_A = REPO_ROOT / "data" / "processed" / "set_a_nofilter.jsonl"
OUT_B = REPO_ROOT / "data" / "processed" / "set_b_magister.jsonl"

ANS_TOL = 1e-6


def _gold_str(parsed: float) -> str:
    """GSM8K answers are always integer-valued; render compactly."""
    return str(int(parsed)) if float(parsed).is_integer() else repr(parsed)


def build():
    OUT_A.parent.mkdir(parents=True, exist_ok=True)

    train = [json.loads(l) for l in GSM8K_TRAIN.open()]
    teacher = json.loads(HO_JSON.read_text())["data"]

    n_total = 0
    n_set_a = 0
    n_set_b = 0
    n_skipped_no_teacher = 0
    n_skipped_unparseable_gold = 0
    n_skipped_unparseable_teacher = 0

    with OUT_A.open("w") as fa, OUT_B.open("w") as fb:
        for i, ex in enumerate(train):
            n_total += 1
            recs = teacher.get(str(i))
            if not recs:
                n_skipped_no_teacher += 1
                continue
            t = recs[0]  # zs_cot has one completion per sample_index

            gold = parse_answer(ex["answer"])
            if gold is None:
                n_skipped_unparseable_gold += 1
                continue

            cot = (t.get("reasoning_completion") or "").strip()
            teacher_pred = parse_answer(t.get("completion"))

            record = {
                "sample_index": i,
                "question": ex["question"],
                "cot": cot,
                "gold_answer": _gold_str(gold),
                "teacher_predicted_answer": (t.get("completion") or "").strip(),
            }
            fa.write(json.dumps(record) + "\n")
            n_set_a += 1

            if teacher_pred is None:
                n_skipped_unparseable_teacher += 1
                continue
            if math.isclose(teacher_pred, gold, abs_tol=ANS_TOL):
                fb.write(json.dumps(record) + "\n")
                n_set_b += 1

    print(f"GSM8K train rows: {n_total}")
    print(f"Set A (no filter): {n_set_a} -> {OUT_A}")
    print(f"Set B (Magister) : {n_set_b} -> {OUT_B}")
    print(f"  skipped: no_teacher={n_skipped_no_teacher} "
          f"unparseable_gold={n_skipped_unparseable_gold} "
          f"unparseable_teacher_pred={n_skipped_unparseable_teacher} (these are still in A)")
    print(f"Set B keep rate (of A): {n_set_b / max(n_set_a, 1):.1%}")

    print("\n--- 3 example records, Set A ---")
    for line in OUT_A.open().readlines()[:3]:
        rec = json.loads(line)
        print(json.dumps({k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                          for k, v in rec.items()}, indent=2))
    print("\n--- 3 example records, Set B ---")
    for line in OUT_B.open().readlines()[:3]:
        rec = json.loads(line)
        print(json.dumps({k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
                          for k, v in rec.items()}, indent=2))


if __name__ == "__main__":
    build()
