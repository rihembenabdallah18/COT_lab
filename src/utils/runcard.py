"""Run-card writer.

A run-card is one JSON file under ``outputs/runs/`` that records what one
stage did: when it ran, how it was configured, what it produced, and the
headline metrics. ``python -m src.status`` reconstructs the project state
from disk by reading every run-card.

Usage:

    from src.utils.runcard import start, finish, fail

    card = start("03_train", "student_set_b", config_dict)
    try:
        ...
    except Exception as e:
        fail(card, f"{type(e).__name__}: {e}")
        raise
    finish(card, metrics={"best_eval_loss": 0.78}, outputs=[ckpt_dir])
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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
