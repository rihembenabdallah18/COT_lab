"""Diagnostic: tokenize Set A and Set B with the FLAN-T5 tokenizer and report
the distribution of input ("Q: {question}") and target ("{cot} #### {gold}")
lengths.

Why: if a large fraction of training targets exceed config.max_target_length
(default 256), they are silently truncated -- the student never sees a complete
"#### {answer}" pattern and learns to keep emitting reasoning steps instead of
terminating. This would explain the looping seen at inference time.

Run:  python -m src.data.audit_lengths
No GPU required.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def percentile(sorted_xs: list[int], q: float) -> int:
    if not sorted_xs:
        return 0
    idx = max(0, min(len(sorted_xs) - 1, int(round(q * (len(sorted_xs) - 1)))))
    return sorted_xs[idx]


def audit_set(name: str, path: Path, tok, cfg: dict) -> None:
    rows = load_jsonl(path)
    n = len(rows)

    inputs = ["Q: " + r["question"] for r in rows]
    targets = [f"{r['cot']} #### {r['gold_answer']}" for r in rows]

    inp_lens = [len(ids) for ids in tok(inputs, add_special_tokens=True).input_ids]
    tgt_lens = [len(ids) for ids in tok(targets, add_special_tokens=True).input_ids]

    inp_sorted = sorted(inp_lens)
    tgt_sorted = sorted(tgt_lens)

    cap_in = cfg["max_input_length"]
    cap_tgt = cfg["max_target_length"]

    n_in_over = sum(1 for x in inp_lens if x > cap_in)
    n_tgt_over_256 = sum(1 for x in tgt_lens if x > 256)
    n_tgt_over_384 = sum(1 for x in tgt_lens if x > 384)
    n_tgt_over_512 = sum(1 for x in tgt_lens if x > 512)

    print(f"\n=== {name}  ({path.name}, n={n}) ===")
    print(f"  inputs (cap={cap_in}):")
    print(f"    p50={percentile(inp_sorted, 0.50)}  "
          f"p90={percentile(inp_sorted, 0.90)}  "
          f"p99={percentile(inp_sorted, 0.99)}  "
          f"max={inp_sorted[-1]}")
    print(f"    > {cap_in} tokens: {n_in_over}/{n} ({n_in_over/n:.1%})")
    print(f"  targets (cap={cap_tgt}):")
    print(f"    p50={percentile(tgt_sorted, 0.50)}  "
          f"p90={percentile(tgt_sorted, 0.90)}  "
          f"p99={percentile(tgt_sorted, 0.99)}  "
          f"max={tgt_sorted[-1]}")
    print(f"    > 256 tokens: {n_tgt_over_256}/{n} ({n_tgt_over_256/n:.1%})  "
          f"<- silently truncated at current config")
    print(f"    > 384 tokens: {n_tgt_over_384}/{n} ({n_tgt_over_384/n:.1%})")
    print(f"    > 512 tokens: {n_tgt_over_512}/{n} ({n_tgt_over_512/n:.1%})")


def main() -> None:
    cfg_path = REPO_ROOT / "config" / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    print(f"loading tokenizer: {cfg['model_name']}")
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])

    for name, key in [("Set A (no filter)", "set_a"), ("Set B (Magister)", "set_b")]:
        audit_set(name, REPO_ROOT / cfg["paths"][key], tok, cfg)


if __name__ == "__main__":
    main()
