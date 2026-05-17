"""Stage 5b — ReCEval scoring for one condition.

Usage:
    python -m src.eval.receval.score_chain --condition student_set_b
    python -m src.eval.receval.score_chain --condition student_set_b --smoke 20
    python -m src.eval.receval.score_chain --condition student_set_b --max-examples 500

Reads outputs/generations/{condition}.jsonl, segments each generated CoT
into steps, and computes three chain-level metrics:

  - intra: min P(entailment | step, step) over steps   (simplified RCU)
  - inter: min [1 − max_r P(contradiction | r, step)] over steps
  - info:  min incremental log p(gold | context) gain over steps

Per-example results -> outputs/eval_results/{condition}_receval.jsonl
After scoring, the script regenerates:
  - outputs/eval_results/receval_summary.csv
  - outputs/plots/receval_violin.png
  - outputs/runs/05b_{condition}.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from tqdm import tqdm

from src.utils.runcard import finish, start

from . import informativeness, inter_step, intra_step
from .segment import segment

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN_DIR = REPO_ROOT / "outputs" / "generations"
EVAL_DIR = REPO_ROOT / "outputs" / "eval_results"
PLOTS_DIR = REPO_ROOT / "outputs" / "plots"

CONDITIONS = [
    "baseline",
    "student_direct_ft",
    "student_set_a",
    "student_set_b",
    "student_set_c",
]

METRICS = ["intra", "inter", "info"]


def _score_example(row: dict, batch_size: int) -> dict:
    question = row["question"]
    cot = row.get("generated_cot") or ""
    gold = row.get("gold_answer")

    steps = segment(cot)
    if not steps:
        steps = [cot.strip()] if cot.strip() else ["(empty)"]

    return {
        "n_steps": len(steps),
        "intra": intra_step.score_chain(steps, batch_size=batch_size),
        "inter": inter_step.score_chain(question, steps, batch_size=batch_size),
        "info": informativeness.score_chain(question, steps, gold),
        "steps": steps,
    }


def _print_detail(row: dict, scored: dict) -> None:
    print(f"\n  Q: {row['question'][:120]}")
    print(f"  gold={row.get('gold_answer')}  n_steps={scored['n_steps']}")
    for i, step in enumerate(scored["steps"]):
        print(f"  step[{i}]: {step[:100]}")
    print(f"  intra={scored['intra']:.4f}  inter={scored['inter']:.4f}"
          f"  info={scored['info']:.4f}")


def _collect_all_results() -> dict[str, list[dict]]:
    by_cond: dict[str, list[dict]] = {}
    for p in sorted(EVAL_DIR.glob("*_receval.jsonl")):
        cond = p.stem.replace("_receval", "")
        with p.open() as f:
            by_cond[cond] = [json.loads(l) for l in f if l.strip()]
    return by_cond


def _summarize(records: list[dict]) -> dict:
    summary: dict = {}
    for metric in METRICS:
        vals = [r[metric] for r in records if not math.isnan(r.get(metric, float("nan")))]
        if not vals:
            for k in ("mean", "std", "min", "max", "n"):
                summary[f"{metric}_{k}"] = 0 if k == "n" else float("nan")
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        summary.update({
            f"{metric}_mean": mean, f"{metric}_std": std,
            f"{metric}_min": min(vals), f"{metric}_max": max(vals),
            f"{metric}_n": len(vals),
        })
    return summary


def _write_summary_csv(by_cond: dict[str, list[dict]]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["condition", "n"]
    for m in METRICS:
        fields += [f"{m}_mean", f"{m}_std", f"{m}_min", f"{m}_max"]
    path = EVAL_DIR / "receval_summary.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for cond in CONDITIONS:
            if cond not in by_cond:
                continue
            s = _summarize(by_cond[cond])
            w.writerow({"condition": cond, "n": len(by_cond[cond]), **s})


def _write_violin(by_cond: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [c for c in CONDITIONS if c in by_cond]
    if not present:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    titles = {"intra": "Intra-step", "inter": "Inter-step", "info": "Informativeness"}
    for ax, metric in zip(axes, METRICS):
        data = [
            [r[metric] for r in by_cond[c] if not math.isnan(r.get(metric, float("nan")))]
            for c in present
        ]
        non_empty = [(d, c) for d, c in zip(data, present) if d]
        if non_empty:
            vdata, vlabels = zip(*non_empty)
            ax.violinplot(list(vdata), showmedians=True)
            ax.set_xticks(range(1, len(vlabels) + 1))
            ax.set_xticklabels(list(vlabels), rotation=25, ha="right", fontsize=8)
        ax.set_title(titles[metric])
        ax.set_ylabel("chain score")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("ReCEval distributions by condition")
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "receval_violin.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--gen-dir", type=Path, default=GEN_DIR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--smoke", type=int, default=0,
                    help="Score only first N examples and print step details.")
    ap.add_argument("--max-examples", type=int, default=0,
                    help="Cap at N examples (use 500 if runtime is tight).")
    ap.add_argument("--info-scorer", default=None,
                    help="Causal LM for informativeness; defaults to config.info_lm.")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
    info_model = args.info_scorer or cfg.get("info_lm") or informativeness._DEFAULT_MODEL
    informativeness.init(info_model)
    print(f"  info scorer: {info_model}")

    gen_path = args.gen_dir / f"{args.condition}.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"No generations file: {gen_path}. Run Stage 4 first.")

    card = start("05b", args.condition, {
        "condition": args.condition,
        "batch_size": args.batch_size,
        "smoke": args.smoke,
        "max_examples": args.max_examples,
        "info_scorer": info_model,
    })

    with gen_path.open() as f:
        lines = [l.strip() for l in f if l.strip()]

    if args.smoke:
        lines = lines[: args.smoke]
    elif args.max_examples:
        lines = lines[: args.max_examples]
    is_subset = bool(args.smoke or (args.max_examples and args.max_examples < 1319))

    results: list[dict] = []
    t0 = time.time()

    for i, line in enumerate(tqdm(lines, desc=args.condition)):
        row = json.loads(line)
        scored = _score_example(row, batch_size=args.batch_size)
        if args.smoke and i < 3:
            _print_detail(row, scored)
        results.append({
            "condition": args.condition,
            "question": row["question"],
            "gold_answer": row.get("gold_answer"),
            "parsed_answer": row.get("parsed_answer"),
            **scored,
        })

    elapsed = time.time() - t0
    per100 = (elapsed / len(results) * 100) if results else 0.0
    print(f"\n  {len(results)} examples  {elapsed:.0f}s total  {per100:.1f}s/100ex")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = EVAL_DIR / f"{args.condition}_receval.jsonl"
    with out_jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    by_cond = _collect_all_results()
    _write_summary_csv(by_cond)
    _write_violin(by_cond)

    s = _summarize(results)
    print(f"  intra={s['intra_mean']:.4f}  inter={s['inter_mean']:.4f}"
          f"  info={s['info_mean']:.4f}")

    notes = []
    if is_subset:
        notes.append(f"subset only: {len(results)} examples")
    if args.smoke:
        notes.append("smoke run — do not include in final results")

    finish(
        card,
        metrics={
            "intra_mean": s["intra_mean"],
            "inter_mean": s["inter_mean"],
            "info_mean": s["info_mean"],
            "intra_std": s["intra_std"],
            "inter_std": s["inter_std"],
            "info_std": s["info_std"],
            "n_scored": len(results),
            "seconds_per_100": per100,
        },
        inputs=[str(gen_path)],
        outputs=[str(out_jsonl),
                 str(EVAL_DIR / "receval_summary.csv"),
                 str(PLOTS_DIR / "receval_violin.png")],
        notes="; ".join(notes),
    )


if __name__ == "__main__":
    main()