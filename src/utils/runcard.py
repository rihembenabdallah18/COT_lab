"""Run-card writer.

A run-card is a single JSON file under ``outputs/runs/`` that records what one
stage actually did: when it ran, how it was configured, what it produced, and
what headline metrics came out. The point is that ``python -m src.status`` can
reconstruct the project state from disk alone, without needing to re-run any
stage.

Usage:

    from src.utils.runcard import start, finish

    card = start("03_train", "student_set_b", config_dict)
    ...
    finish(card, metrics={"best_eval_loss": 0.78}, outputs=[ckpt_dir],
           samples=[{"input": q, "output": pred, "gold": gold}])

`finish` writes the card atomically (write to .tmp then rename).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "outputs" / "runs"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_config(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


def _card_path(stage: str, run_name: str) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR / f"{stage}_{run_name}.json"


def start(stage: str, run_name: str, config: dict | None = None) -> dict:
    """Open a run-card. Returns the in-memory dict; finish() persists it."""
    cfg = config or {}
    return {
        "stage": stage,
        "run_name": run_name,
        "started_at": _utcnow_iso(),
        "completed_at": None,
        "duration_seconds": None,
        "status": "running",
        "config_hash": _hash_config(cfg),
        "config_snapshot": cfg,
        "inputs": [],
        "outputs": [],
        "metrics": {},
        "samples": [],
        "notes": "",
        "_t0": time.time(),
    }


def finish(
    card: dict,
    metrics: dict | None = None,
    inputs: Iterable[str] | None = None,
    outputs: Iterable[str] | None = None,
    samples: list[dict] | None = None,
    notes: str = "",
    status: str = "completed",
) -> Path:
    """Close out a run-card and write it atomically to disk."""
    card["completed_at"] = _utcnow_iso()
    card["duration_seconds"] = round(time.time() - card.pop("_t0", time.time()), 2)
    card["status"] = status
    if metrics:
        card["metrics"].update(metrics)
    if inputs:
        card["inputs"] = [str(p) for p in inputs]
    if outputs:
        card["outputs"] = [str(p) for p in outputs]
    if samples:
        card["samples"] = samples
    if notes:
        card["notes"] = notes

    out = _card_path(card["stage"], card["run_name"])
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(card, f, indent=2, default=str)
    os.replace(tmp, out)
    return out


def fail(card: dict, error: str) -> Path:
    return finish(card, status="failed", notes=error)


def append_event(run_name: str, stage: str, event: dict) -> None:
    """Append a JSONL progress event next to the run-card.

    Long-running stages (training, ReCEval) call this on every epoch / batch
    boundary so a plot script can read progress without pickling Trainer state.
    """
    progress_dir = RUNS_DIR / f"{stage}_{run_name}_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    log = progress_dir / "progress.jsonl"
    event = {"t": _utcnow_iso(), **event}
    with log.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def load_all() -> list[dict]:
    """Return every run-card on disk, sorted by (stage, run_name)."""
    if not RUNS_DIR.exists():
        return []
    cards: list[dict] = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        try:
            cards.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return cards
