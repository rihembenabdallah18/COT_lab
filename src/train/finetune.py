"""Stage 3: fine-tune FLAN-T5-base on a JSONL training set.

Input format : "Q: {question}"
Target format: "{cot} #### {gold_answer}"  (collapses to "#### {ans}" when cot is empty)

Holds out a 10% deterministic validation slice (seed=42). Saves a checkpoint
per epoch to outputs/checkpoints/{run_name}/. Resumable via --resume.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

from src.utils.runcard import fail, finish, start

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fmt_time(seconds: float) -> str:
    s = max(int(seconds), 0)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


class EpochLogger(TrainerCallback):
    """One clean line per epoch: train / val loss, elapsed, ETA."""

    def __init__(self, run_name: str, total_epochs: int):
        self.run_name = run_name
        self.total = total_epochs
        self.t0: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        print(f"[{self.run_name}] training started "
              f"({self.total} epochs, {state.max_steps} steps)", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(round(state.epoch)) if state.epoch else 0
        elapsed = time.time() - (self.t0 or time.time())
        eta = (elapsed / max(epoch, 1)) * max(self.total - epoch, 0)
        train_loss = next(
            (e["loss"] for e in reversed(state.log_history)
             if "loss" in e and "eval_loss" not in e),
            None,
        )
        val_loss = (metrics or {}).get("eval_loss")
        parts = [f"[{self.run_name}] epoch {epoch}/{self.total}"]
        if train_loss is not None:
            parts.append(f"train={train_loss:.4f}")
        if val_loss is not None:
            parts.append(f"val={val_loss:.4f}")
        parts.append(f"elapsed={_fmt_time(elapsed)}")
        parts.append(f"eta={_fmt_time(eta)}")
        print("  ".join(parts), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - (self.t0 or time.time())
        print(f"[{self.run_name}] training finished in {_fmt_time(elapsed)}",
              flush=True)


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def build_trainer(cfg: dict, run_dir: Path, ds_train, ds_val, n_epochs: int,
                  run_name: str):
    model_name = cfg.get("active_model_name") or cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize(batch):
        inputs = ["Q: " + q for q in batch["question"]]
        targets = [
            (f"{cot} #### {ans}" if cot else f"#### {ans}")
            for cot, ans in zip(batch["cot"], batch["gold_answer"])
        ]
        x = tok(inputs, max_length=cfg["max_input_length"], truncation=True)
        y = tok(targets, max_length=cfg["max_target_length"], truncation=True)
        x["labels"] = y["input_ids"]
        return x

    ds_train_t = ds_train.map(tokenize, batched=True, remove_columns=ds_train.column_names)
    ds_val_t = ds_val.map(tokenize, batched=True, remove_columns=ds_val.column_names)

    use_cuda = torch.cuda.is_available()
    fp16 = bool(cfg.get("fp16", False)) and use_cuda

    targs = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=n_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        fp16=fp16,
        seed=cfg["seed"],
        predict_with_generate=False,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type="cosine",
    )

    callbacks: list[TrainerCallback] = [EpochLogger(run_name, n_epochs)]
    patience = int(cfg.get("early_stopping_patience", 0) or 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds_train_t,
        eval_dataset=ds_val_t,
        tokenizer=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
        callbacks=callbacks,
    )
    return trainer, fp16, use_cuda


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--train", required=True, help="JSONL training set (relative to repo root)")
    p.add_argument("--run-name", required=True, help="run name; outputs go to ckpt_root/run_name")
    p.add_argument("--limit", type=int, default=None, help="Truncate training set (smoke runs)")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--model", default=None,
                   help="Override config.model_name (e.g. google/flan-t5-small)")
    args = p.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    set_seed(cfg["seed"])
    cfg["active_model_name"] = args.model or cfg["model_name"]

    run_dir = REPO_ROOT / cfg["paths"]["ckpt_root"] / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(REPO_ROOT / args.train)
    if args.limit:
        rows = rows[: args.limit]

    full = Dataset.from_list(rows)
    splits = full.train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    ds_train, ds_val = splits["train"], splits["test"]
    print(f"[data] train={len(ds_train)} val={len(ds_val)} (from {len(rows)} rows)")

    n_epochs = args.epochs or cfg["num_epochs"]

    card = start("03_train", args.run_name, {
        "model_name": cfg["active_model_name"],
        "train_file": args.train,
        "n_epochs": n_epochs,
        "limit": args.limit,
        "learning_rate": cfg["learning_rate"],
        "weight_decay": cfg["weight_decay"],
        "warmup_ratio": cfg["warmup_ratio"],
        "batch_size": cfg["batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "max_input_length": cfg["max_input_length"],
        "max_target_length": cfg["max_target_length"],
        "early_stopping_patience": cfg.get("early_stopping_patience"),
        "seed": cfg["seed"],
    })

    try:
        trainer, fp16, use_cuda = build_trainer(
            cfg, run_dir, ds_train, ds_val, n_epochs, args.run_name)
        print(f"[device] cuda={use_cuda} fp16={fp16}")
        trainer.train(resume_from_checkpoint=True if args.resume else None)
    except Exception as e:
        fail(card, f"{type(e).__name__}: {e}")
        raise

    best = None
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and (best is None or entry["eval_loss"] < best["eval_loss"]):
            best = entry

    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    finish(
        card,
        metrics={
            "n_train": len(ds_train),
            "n_val": len(ds_val),
            "n_epochs_completed": trainer.state.epoch,
            "best_epoch": best.get("epoch") if best else None,
            "best_eval_loss": best.get("eval_loss") if best else None,
            "device": "cuda" if use_cuda else "cpu",
            "fp16": fp16,
        },
        inputs=[args.train],
        outputs=[str(c.relative_to(REPO_ROOT)) for c in ckpts],
        notes=(f"lr={cfg['learning_rate']}, wd={cfg['weight_decay']}, "
               f"epochs<= {n_epochs}, patience={cfg.get('early_stopping_patience')}"),
    )


if __name__ == "__main__":
    main()
