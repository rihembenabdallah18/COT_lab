"""Project-wide status command.

Run::

    python -m src.status              # human-readable table
    python -m src.status --json       # machine-readable

Reads every run-card under ``outputs/runs/`` plus a few file-presence checks
that don't have run-cards yet (e.g. raw data) and prints a one-screen view of
where every stage stands. Designed so that the project state is readable from
disk alone, no live process needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.runcard import load_all

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _headline(card: dict) -> str:
    """One-line summary derived from the card's metrics. Stage-specific."""
    m = card.get("metrics") or {}
    stage = card.get("stage", "")
    run = card.get("run_name", "")
    key = f"{stage}_{run}" if run else stage
    if key.startswith("01"):
        return f"gsm8k_train={m.get('gsm8k_train', '?')} gsm8k_test={m.get('gsm8k_test', '?')} ho={m.get('ho_records', '?')}"
    if key.startswith("02"):
        return (f"A={m.get('set_a_size', '?')} B={m.get('set_b_size', '?')} "
                f"C={m.get('set_c_size', '?')} direct={m.get('direct_ft_size', '?')}")
    if key.startswith("03"):
        be = m.get("best_eval_loss")
        ep = m.get("best_epoch")
        if be is None:
            return "no eval loss yet"
        return f"best_eval_loss={be:.3f} @ epoch {ep}"
    if key.startswith("04"):
        n = m.get("n_generated")
        sps = m.get("seconds_per_example")
        return f"n={n} ({sps:.1f}s/ex)" if (n is not None and sps is not None) else "?"
    if key.startswith("05a"):
        return ", ".join(f"{c}={v:.2%}" for c, v in (m.get("acc_per_condition") or {}).items())
    if key.startswith("05b"):
        return (f"intra={m.get('intra_mean', float('nan')):.2f} "
                f"inter={m.get('inter_mean', float('nan')):.2f} "
                f"info={m.get('info_mean', float('nan')):.2f}")
    return ""


def _print_table(cards: list[dict]) -> None:
    if not cards:
        print("No run-cards found under outputs/runs/. Run a stage first.")
        return
    rows = []
    for c in cards:
        rows.append((
            c.get("stage", "?"),
            c.get("run_name", "?"),
            c.get("status", "?"),
            _fmt_duration(c.get("duration_seconds")),
            _headline(c),
        ))
    widths = [max(len(str(r[i])) for r in rows) for i in range(5)]
    headers = ("Stage", "Run", "Status", "Duration", "Headline")
    widths = [max(widths[i], len(headers[i])) for i in range(5)]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


def _check_file_presence() -> list[dict]:
    """Soft pre-flight: report a few key paths even if they have no run-card."""
    checks = [
        ("data/raw/gsm8k/train.jsonl", "GSM8K train"),
        ("data/raw/gsm8k/test.jsonl", "GSM8K test"),
        ("data/raw/ho_et_al_cots/gsm8k_zs_cot_text-davinci-002.json", "Ho et al. CoTs"),
        ("data/processed/set_a_nofilter.jsonl", "Set A"),
        ("data/processed/set_b_magister.jsonl", "Set B"),
        ("data/processed/set_c_calculator.jsonl", "Set C"),
        ("data/processed/direct_ft.jsonl", "Direct FT"),
    ]
    out = []
    for rel, label in checks:
        p = REPO_ROOT / rel
        out.append({"path": rel, "label": label, "exists": p.exists(),
                    "size_bytes": p.stat().st_size if p.exists() else 0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = ap.parse_args()

    cards = load_all()
    files = _check_file_presence()

    if args.json:
        print(json.dumps({"cards": cards, "files": files}, indent=2, default=str))
        return

    print("=" * 78)
    print("  COT_lab project status")
    print("=" * 78)
    _print_table(cards)
    print()
    print("Key files:")
    for f in files:
        flag = "OK " if f["exists"] else "-- "
        size = f"{f['size_bytes'] / 1e6:.1f} MB" if f["exists"] else "missing"
        print(f"  [{flag}] {f['label']:<18} {f['path']:<60} {size}")


if __name__ == "__main__":
    main()
