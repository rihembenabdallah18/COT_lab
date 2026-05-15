"""Stage 5a — accuracy and accuracy-with-calculator.

For each condition with a Stage-4 generations JSONL, parse the predicted
final answer (#### priority, last-number fallback) and
compare to gold (tolerance abs(pred - gold) < 1e-6).

The ``with_calc`` variant runs ``correct_and_propagate`` on the generated
CoT first, then re-parses. Only applied to CoT student conditions
(student_set_*) where it is meaningful. Skipped for baseline (harmful —
free-form text has no real equations) and direct_ft (no-op — emits only
#### N with no intermediate equations).

Outputs:
  - outputs/eval_results/accuracy.csv (condition, n, correct, accuracy,
    correct_w_calc, accuracy_w_calc)
  - outputs/plots/accuracy_bar.png (two bars per condition)
  - outputs/runs/05a_accuracy.json (run-card consumed by `python -m src.status`)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.calculator import correct_and_propagate
from src.data.parse_answer import parse_answer
from src.utils.runcard import finish, start

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = REPO_ROOT / "outputs" / "generations"
EVAL_DIR = REPO_ROOT / "outputs" / "eval_results"
PLOTS_DIR = REPO_ROOT / "outputs" / "plots"

DEFAULT_CONDITIONS = [
    "baseline",
    "student_direct_ft",
    "student_set_a",
    "student_set_b",
    "student_set_c",
    "student_set_a_oc",
    "student_set_b_oc",
    "student_set_c_oc",
]

TOL = 1e-6


def _equal(pred: float | None, gold: float | None) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < TOL


def _score_file(path: Path, run_calc: bool = False) -> dict:
    n = correct = correct_w_calc = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cot = row.get("generated_cot") or ""
            gold = row.get("gold_answer")
            if isinstance(gold, str):
                gold = parse_answer(gold)

            pred = parse_answer(cot)
            n += 1
            if _equal(pred, gold):
                correct += 1

            if run_calc:
                propagated_cot, _ = correct_and_propagate(cot)
                if _equal(parse_answer(propagated_cot), gold):
                    correct_w_calc += 1

    acc = correct / n if n else 0.0
    acc_w_calc = correct_w_calc / n if n else 0.0
    return {
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "correct_w_calc": correct_w_calc if run_calc else None,
        "accuracy_w_calc": acc_w_calc if run_calc else None,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "n", "correct", "accuracy", "correct_w_calc", "accuracy_w_calc"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def _plot_bar(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [r["condition"] for r in rows]
    acc = [r["accuracy"] * 100 for r in rows]
    acc_w = [r["accuracy_w_calc"] * 100 if r["accuracy_w_calc"] is not None else 0
             for r in rows]

    x = range(len(conds))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - width / 2 for i in x], acc, width=width, label="accuracy")
    ax.bar([i + width / 2 for i in x], acc_w, width=width, label="accuracy_w_calc")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conds, rotation=25, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("GSM8K test accuracy by condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _print_table(rows: list[dict]) -> None:
    headers = ("condition", "n", "accuracy", "accuracy_w_calc")
    col_w = [max(len(h), 12) for h in headers]
    col_w[0] = max(col_w[0], max(len(r["condition"]) for r in rows))
    col_w[1] = max(col_w[1], max(len(str(r["n"])) for r in rows))
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(col_w))
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_w))
    for r in rows:
        w_calc = f"{r['accuracy_w_calc']:.2%}" if r["accuracy_w_calc"] is not None else "  n/a  "
        print(fmt.format(r["condition"], r["n"], f"{r['accuracy']:.2%}", w_calc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--gen-dir", type=Path, default=GEN_DIR)
    ap.add_argument("--out-csv", type=Path, default=EVAL_DIR / "accuracy.csv")
    ap.add_argument("--plot", type=Path, default=PLOTS_DIR / "accuracy_bar.png")
    args = ap.parse_args()

    card = start("05a", "accuracy", {
        "conditions": args.conditions,
        "gen_dir": str(args.gen_dir),
        "tolerance": TOL,
    })

    rows: list[dict] = []
    inputs: list[str] = []
    missing: list[str] = []
    for cond in args.conditions:
        path = args.gen_dir / f"{cond}.jsonl"
        if not path.exists():
            missing.append(cond)
            continue
        inputs.append(str(path))
        run_calc = cond.startswith("student_set_") and not cond.endswith("_oc")
        scored = _score_file(path, run_calc=run_calc)
        rows.append({"condition": cond, **scored})

    if not rows:
        finish(card, status="failed",
               notes=f"no generations found in {args.gen_dir}; missing: {missing}")
        raise SystemExit(f"No generation files found in {args.gen_dir}. Run Stage 4 first.")

    _write_csv(rows, args.out_csv)
    _plot_bar(rows, args.plot)
    _print_table(rows)

    acc_per_condition = {r["condition"]: r["accuracy"] for r in rows}
    acc_w_calc_per_condition = {r["condition"]: r["accuracy_w_calc"] for r in rows
                                if r["accuracy_w_calc"] is not None}

    baseline_acc = acc_per_condition.get("baseline")
    below_baseline: list[str] = []
    if baseline_acc is not None:
        for cond, a in acc_per_condition.items():
            if cond.startswith("student_") and a < baseline_acc:
                below_baseline.append(cond)

    notes_parts = []
    if missing:
        notes_parts.append(f"missing conditions: {missing}")
    if below_baseline:
        notes_parts.append(
            "STOP per AGENT.md §5a — distilled student(s) below baseline: "
            + ", ".join(below_baseline)
        )
    notes = "; ".join(notes_parts)

    finish(
        card,
        metrics={
            "acc_per_condition": acc_per_condition,
            "acc_w_calc_per_condition": acc_w_calc_per_condition,
            "n_conditions_scored": len(rows),
            "baseline_acc": baseline_acc,
            "below_baseline_students": below_baseline,
        },
        inputs=inputs,
        outputs=[str(args.out_csv), str(args.plot)],
        notes=notes,
    )

    if below_baseline:
        print()
        print(f"!! STOP: distilled student(s) below baseline ({baseline_acc:.2%}): "
              f"{below_baseline}")
        print("   Per AGENT.md §5a do not proceed to ReCEval — recipe is still broken.")
    elif missing:
        print()
        print(f"!! Note: scored {len(rows)} of {len(args.conditions)} conditions; "
              f"missing generations for: {missing}")


if __name__ == "__main__":
    main()
