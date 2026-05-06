"""Stage 4: greedy inference on the GSM8K test set.

Loads a model (FLAN-T5-base for baseline, or a fine-tuned checkpoint for the
students), runs greedy decoding on all 1,319 test examples, and writes JSONL
records to outputs/generations/{condition}.jsonl.

Resumable: already-written records are detected by line count and skipped;
the script appends only missing examples on restart.
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

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def run_inference(
    model_path: str,
    condition: str,
    cfg: dict,
    test_path: Path,
    out_path: Path,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int = 0,
) -> None:
    test_data = load_jsonl(test_path)
    n_total = len(test_data)

    already_done = _count_lines(out_path)
    if already_done >= n_total:
        print(f"[{condition}] already complete ({already_done}/{n_total}), skipping.")
        return

    print(f"[{condition}] loading model from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"[{condition}] device={device}, resuming from record {already_done}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = cfg.get("inference_batch_size", 8)
    max_new_tokens = cfg.get("inference_max_new_tokens", 256)

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
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            if repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = repetition_penalty
            if no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
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
    print(f"[{condition}] done: {n_generated} examples in {elapsed:.0f}s "
          f"({elapsed/max(n_generated,1):.1f}s/ex) -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument(
        "--condition",
        required=True,
        choices=["baseline", "student_set_a", "student_set_b"],
        help="Which condition to run",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to HF checkpoint dir (required for student conditions). "
             "If omitted for a student condition, the latest checkpoint under "
             "outputs/checkpoints/{run_name}/ is used automatically.",
    )
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="If set, passes repetition_penalty to model.generate (e.g. 1.3).",
    )
    p.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="If >0, blocks repetition of n-grams of this size during decoding.",
    )
    p.add_argument(
        "--out-suffix",
        default="",
        help="Suffix appended to output filename: {condition}{suffix}.jsonl. "
             "Use to keep diagnostic re-runs from clobbering original generations.",
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
        run_name = args.condition  # "student_set_a" or "student_set_b"
        run_dir = REPO_ROOT / cfg["paths"]["ckpt_root"] / run_name
        ckpt = _best_checkpoint(run_dir)
        model_path = str(ckpt)
        print(f"[auto] using checkpoint: {ckpt.name}")

    out_path = gen_dir / f"{args.condition}{args.out_suffix}.jsonl"
    run_inference(
        model_path,
        args.condition,
        cfg,
        test_path,
        out_path,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )


if __name__ == "__main__":
    main()
