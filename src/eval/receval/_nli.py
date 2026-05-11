"""Shared DeBERTa-v3 NLI model singleton.

Loaded lazily on first call; placed on GPU if available.
Both intra_step and inter_step import from here to avoid loading two copies.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_state: dict = {}


def _load() -> None:
    if "model" in _state:
        return
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(device).eval()
    label_map = {lbl.lower(): int(idx) for idx, lbl in mdl.config.id2label.items()}
    _state.update(model=mdl, tokenizer=tok, device=device, label_map=label_map)


def label_names() -> dict[str, int]:
    _load()
    return dict(_state["label_map"])


@torch.no_grad()
def batch_probs(
    pairs: list[tuple[str, str]],
    label: str,
    batch_size: int = 16,
) -> list[float]:
    """Return P(label) for each (premise, hypothesis) pair."""
    _load()
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    device = _state["device"]
    label_map = _state["label_map"]
    if label not in label_map:
        raise ValueError(f"Label '{label}' not in model labels: {label_map}")
    idx = label_map[label]

    result: list[float] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        premises, hypotheses = zip(*chunk)
        enc = tokenizer(
            list(premises),
            list(hypotheses),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, idx]
        result.extend(probs.cpu().tolist())
    return result
