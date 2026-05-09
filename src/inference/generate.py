"""Stage 4: inference on the GSM8K test set.

Loads a model (FLAN-T5-base for baseline, or a fine-tuned checkpoint for the
students), runs decoding with the v2 recipe (beam=4 + no_repeat_ngram_size=4
+ repetition_penalty=1.15 by default, all overridable from CLI), and writes
JSONL records to outputs/generations/{condition}.jsonl.

Defaults come from config/config.yaml so the same script handles every
condition without per-condition flag plumbing in scripts/04_inference.sh.
v1 used pure greedy and looped; never restore that default.

Resumable: already-written records are detected by line count and skipped.
Writes a Stage-4 run-card per condition to outputs/runs/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.parse_answer import parse_answer
from src.utils.runcard import fail, finish, start

REPO_ROOT = Path(__file__).resolve().parents[2]

# All currently-supported conditions. Each maps to either a HF model name
# (baseline) or a checkpoint dir under outputs/checkpoints/{run_name}/.
CONDITIONS = ["baseline", "student_direct_ft",
              "student_set_a", "student_set_b", "student_set_c"]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def _best_checkpoint(run_dir: Path) -> Path:
    """Return the checkpoint-{step} sub-dir with the highest step number."""
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return ckpts[-1]


def _load_tokenizer(model_path: str, fallback: str):
    """Load tokenizer from the checkpoint, then run-dir, then base model.

    HF Trainer with save_total_limit=2 + load_best_model_at_end=True is known
    to skip writing tokenizer files into rotated checkpoint dirs in some
    transformers versions. The tokenizer is identical to the base model
    (we never modify the vocab), so falling back to that is safe.
    """
    candidates = [model_path]
    parent = str(Path(model_path).parent)
    if parent != model_path:
        candidates.append(parent)
    candidates.append(fallback)
    last_err: Exception | None = None
    for cand in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            if cand != model_path:
                print(f"[tokenizer] not in {model_path}; loaded from {cand}")
            return tok
        except (OSError, ValueError) as e:
            last_err = e
    raise RuntimeError(
        f"could not load tokenizer from any of {candidates}; last err: {last_err}"
    )


def _build_gen_kwargs(cfg: dict, args) -> dict:
    """v2 decoding defaults read from config; CLI overrides win."""
    num_beams = args.num_beams if args.num_beams is not None \
        else cfg.get("inference_num_beams", 4)
    rep_penalty = args.repetition_penalty if args.repetition_penalty is not None \
        else cfg.get("inference_repetition_penalty", 1.15)
    no_rep = args.no_repeat_ngram_size if args.no_repeat_ngram_size is not None \
        else cfg.get("inference_no_repeat_ngram_size", 4)
    length_penalty = cfg.get("inference_length_penalty", 1.0)
    max_new = args.max_new_tokens if args.max_new_tokens is not None \
        else cfg.get("inference_max_new_tokens", 512)

    gk = {
        "max_new_tokens": max_new,
        "do_sample": False,
        "num_beams": num_beams,
        "length_penalty": length_penalty,
    }
    if rep_penalty and rep_penalty != 1.0:
        gk["repetition_penalty"] = rep_penalty
    if no_rep and no_rep > 0:
        gk["no_repeat_ngram_size"] = no_rep
    return gk


def run_inference(
    model_path: str,
    condition: str,
    cfg: dict,
    test_path: Path,
    out_path: Path,
    args,
) -> dict:
    test_data = load_jsonl(test_path)
    n_total = len(test_data)
    gen_kwargs = _build_gen_kwargs(cfg, args)

    already_done = _count_lines(out_path)
    if already_done >= n_total:
        print(f"[{condition}] already complete ({already_done}/{n_total}), skipping.")
        return {
            "n_total": n_total,
            "n_generated": 0,
            "already_done": already_done,
            "seconds_per_example": None,
            "gen_kwargs": gen_kwargs,
        }

    print(f"[{condition}] loading model from {model_path}")
    tok = _load_tokenizer(model_path, fallback=cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"[{condition}] device={device}, resuming from record {already_done}")
    print(f"[{condition}] gen_kwargs={gen_kwargs}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = cfg.get("inference_batch_size", 8)

    todo = test_data[already_done:]
    t_start = time.time()

    with out_path.open("a") as fout:
        for batch_start in tqdm(range(0, len(todo), batch_size), desc=condition):
            batch = todo[batch_start: batch_start + batch_size]
            inputs = ["Q: " + ex["question"] for ex in batch]
            enc = tok(
                inputs,
                max_length=cfg["max_input_length"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(**enc, **gen_kwargs)
            for ex, ids in zip(batch, out_ids):
                generated = tok.decode(ids, skip_special_tokens=True)
                gold = parse_answer(ex["answer"])
                record = {
                    "question": ex["question"],
                    "generated_cot": generated,
                    "parsed_answer": parse_answer(generated),
                    "gold_answer": gold,
                }
                fout.write(json.dumps(record) + "\n")

    elapsed = time.time() - t_start
    n_generated = len(todo)
    sec_per = elapsed / max(n_generated, 1)
    print(f"[{condition}] done: {n_generated} examples in {elapsed:.0f}s "
          f"({sec_per:.1f}s/ex) -> {out_path}")
    return {
        "n_total": n_total,
        "n_generated": n_generated,
        "already_done": already_done,
        "seconds_per_example": round(sec_per, 3),
        "gen_kwargs": gen_kwargs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument(
        "--condition",
        required=True,
        choices=CONDITIONS,
        help="Which condition to run",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to HF checkpoint dir. If omitted for a student condition, "
             "the latest checkpoint under outputs/checkpoints/{condition}/ "
             "is used automatically.",
    )
    p.add_argument("--num-beams", type=int, default=None,
                   help="Override config.inference_num_beams (default 4).")
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="Override config.inference_repetition_penalty "
                        "(default 1.15). Pass 1.0 to disable.")
    p.add_argument("--no-repeat-ngram-size", type=int, default=None,
                   help="Override config.inference_no_repeat_ngram_size "
                        "(default 4). Pass 0 to disable.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override config.inference_max_new_tokens (default 512).")
    p.add_argument(
        "--out-suffix",
        default="",
        help="Suffix appended to output filename: {condition}{suffix}.jsonl. "
             "Use to keep diagnostic re-runs from clobbering originals.",
    )
    args = p.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    test_path = REPO_ROOT / cfg["paths"]["gsm8k_test"]
    gen_dir = REPO_ROOT / cfg["paths"]["generations_dir"]

    if args.condition == "baseline":
        model_path = cfg["model_name"]
    elif args.checkpoint:
        model_path = args.checkpoint
    else:
        run_dir = REPO_ROOT / cfg["paths"]["ckpt_root"] / args.condition
        ckpt = _best_checkpoint(run_dir)
        model_path = str(ckpt)
        print(f"[auto] using checkpoint: {ckpt.name}")

    out_path = gen_dir / f"{args.condition}{args.out_suffix}.jsonl"

    card = start("04_inference", args.condition, {
        "model_path": str(model_path),
        "out_suffix": args.out_suffix,
        "config_keys": {k: cfg.get(k) for k in (
            "inference_num_beams", "inference_no_repeat_ngram_size",
            "inference_repetition_penalty", "inference_length_penalty",
            "inference_max_new_tokens", "inference_batch_size",
            "max_input_length",
        )},
        "cli_overrides": {
            "num_beams": args.num_beams,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_new_tokens": args.max_new_tokens,
        },
    })

    try:
        result = run_inference(model_path, args.condition, cfg,
                               test_path, out_path, args)
    except Exception as e:
        fail(card, f"{type(e).__name__}: {e}")
        raise

    samples = []
    if out_path.exists():
        with out_path.open() as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                rec = json.loads(line)
                rec["generated_cot"] = rec["generated_cot"][:300]
                samples.append(rec)

    finish(
        card,
        metrics={
            "n_total": result["n_total"],
            "n_generated": result["n_generated"],
            "already_done": result["already_done"],
            "seconds_per_example": result["seconds_per_example"],
        },
        inputs=[str(test_path.relative_to(REPO_ROOT))],
        outputs=[str(out_path.relative_to(REPO_ROOT))],
        samples=samples,
        notes=f"gen_kwargs={result['gen_kwargs']}",
    )


if __name__ == "__main__":
    main()
