"""Stage 4: inference on the GSM8K test set.

Loads a model (FLAN-T5-base for baseline, or a fine-tuned checkpoint for
a student), runs beam decoding with the recipe from config/config.yaml, and
writes JSONL records to outputs/generations/{condition}.jsonl.

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

CONDITIONS = [
    "baseline",
    "student_direct_ft",
    "student_set_a",
    "student_set_b",
    "student_set_c",
]


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
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return ckpts[-1]


def _build_gen_kwargs(cfg: dict) -> dict:
    gk = {
        "max_new_tokens": cfg["inference_max_new_tokens"],
        "do_sample": False,
        "num_beams": cfg["inference_num_beams"],
        "length_penalty": cfg.get("inference_length_penalty", 1.0),
    }
    if cfg.get("inference_repetition_penalty", 1.0) != 1.0:
        gk["repetition_penalty"] = cfg["inference_repetition_penalty"]
    if cfg.get("inference_no_repeat_ngram_size", 0) > 0:
        gk["no_repeat_ngram_size"] = cfg["inference_no_repeat_ngram_size"]
    return gk


def run_inference(
    model_path: str,
    condition: str,
    cfg: dict,
    test_path: Path,
    out_path: Path,
) -> dict:
    test_data = load_jsonl(test_path)
    n_total = len(test_data)
    gen_kwargs = _build_gen_kwargs(cfg)

    already_done = _count_lines(out_path)
    if already_done >= n_total:
        print(f"[{condition}] already complete ({already_done}/{n_total}), skipping.")
        return {"n_total": n_total, "n_generated": 0,
                "already_done": already_done, "seconds_per_example": None,
                "gen_kwargs": gen_kwargs}

    print(f"[{condition}] loading model from {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[{condition}] device={device}, resuming from record {already_done}")
    print(f"[{condition}] gen_kwargs={gen_kwargs}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = cfg["inference_batch_size"]
    todo = test_data[already_done:]
    t0 = time.time()

    with out_path.open("a") as fout:
        for start_idx in tqdm(range(0, len(todo), batch_size), desc=condition):
            batch = todo[start_idx: start_idx + batch_size]
            inputs = ["Q: " + ex["question"] for ex in batch]
            enc = tok(inputs, max_length=cfg["max_input_length"],
                      truncation=True, padding=True,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                out_ids = model.generate(**enc, **gen_kwargs)
            for ex, ids in zip(batch, out_ids):
                generated = tok.decode(ids, skip_special_tokens=True)
                record = {
                    "question": ex["question"],
                    "generated_cot": generated,
                    "parsed_answer": parse_answer(generated),
                    "gold_answer": parse_answer(ex["answer"]),
                }
                fout.write(json.dumps(record) + "\n")

    elapsed = time.time() - t0
    sec_per = elapsed / max(len(todo), 1)
    print(f"[{condition}] done: {len(todo)} examples in {elapsed:.0f}s "
          f"({sec_per:.2f}s/ex) -> {out_path}")
    return {"n_total": n_total, "n_generated": len(todo),
            "already_done": already_done,
            "seconds_per_example": round(sec_per, 3),
            "gen_kwargs": gen_kwargs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (defaults to latest under "
                        "outputs/checkpoints/{condition}/)")
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
        model_path = str(_best_checkpoint(run_dir))
        print(f"[auto] using checkpoint: {model_path}")

    out_path = gen_dir / f"{args.condition}.jsonl"
    card = start("04_inference", args.condition, {
        "model_path": str(model_path),
        "num_beams": cfg["inference_num_beams"],
        "max_new_tokens": cfg["inference_max_new_tokens"],
        "repetition_penalty": cfg["inference_repetition_penalty"],
        "no_repeat_ngram_size": cfg["inference_no_repeat_ngram_size"],
    })

    try:
        result = run_inference(model_path, args.condition, cfg, test_path, out_path)
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
