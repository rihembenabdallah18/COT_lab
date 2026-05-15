"""Stage 3: fine-tune FLAN-T5-base on a JSONL training set.

Input format : "Q: {question}"
Target format: "{cot} #### {gold_answer}"

Holds out a 10% deterministic validation slice (seed=42). Saves a checkpoint
per epoch to `outputs/checkpoints/{run_name}/`. Resumable via --resume.
Logs train and eval loss to `loss_log.csv` in the same directory.

Note: HF Trainer names checkpoints `checkpoint-{step}`, not `epoch_{N}`.
The training_summary.json identifies the best-validation-loss checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
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

from src.utils.runcard import append_event, fail, finish, start

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


class CSVLossLogger(TrainerCallback):
    """Append train/eval loss rows to a CSV on every Trainer log."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(["step", "epoch", "kind", "loss"])

    def _append(self, step, epoch, kind, loss):
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([step, epoch, kind, loss])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self._append(state.global_step, state.epoch, "train", logs["loss"])
        if "eval_loss" in logs:
            self._append(state.global_step, state.epoch, "eval", logs["eval_loss"])


class RunCardProgressLogger(TrainerCallback):
    """Mirror per-epoch eval losses into the stage's progress.jsonl so
    `python -m src.status` and the plot helpers can read training progress
    without poking into HF Trainer state.
    """

    def __init__(self, stage: str, run_name: str):
        self.stage = stage
        self.run_name = run_name

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "eval_loss" in logs:
            append_event(self.run_name, self.stage, {
                "event": "eval",
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_loss": logs["eval_loss"],
            })
        elif "loss" in logs:
            append_event(self.run_name, self.stage, {
                "event": "train",
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs["loss"],
            })


def build_trainer(cfg: dict, run_dir: Path, ds_train, ds_val, n_epochs: int,
                  run_name: str):
    model_name = cfg.get("active_model_name") or cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize(batch):
        inputs = ["Q: " + q for q in batch["question"]]
        # Direct FT records carry cot="" so the target collapses to ` #### {ans}`.
        # We strip a leading separator-only target to keep formatting tidy.
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
        save_total_limit=2,                 # best + most recent (early-stop friendly)
        logging_steps=10,
        fp16=fp16,
        seed=cfg["seed"],
        predict_with_generate=False,
        report_to=[],
        load_best_model_at_end=True,        # required for EarlyStoppingCallback
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_smoothing_factor=0.1,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
    )

    collator = DataCollatorForSeq2Seq(tok, model=model)

    callbacks: list[TrainerCallback] = [
        CSVLossLogger(run_dir / "loss_log.csv"),
        RunCardProgressLogger("03_train", run_name),
    ]
    patience = int(cfg.get("early_stopping_patience", 0) or 0)
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=ds_train_t,
        eval_dataset=ds_val_t,
        tokenizer=tok,
        data_collator=collator,
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
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in run dir")
    p.add_argument("--model", default=None,
                   help="Override config.model_name (e.g. google/flan-t5-small for size ablation)")
    args = p.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    set_seed(cfg["seed"])
    if args.model:
        cfg["active_model_name"] = args.model
    else:
        cfg["active_model_name"] = cfg["model_name"]

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

        resume_arg = True if args.resume else None
        trainer.train(resume_from_checkpoint=resume_arg)
    except Exception as e:
        fail(card, f"{type(e).__name__}: {e}")
        raise

    # Identify best checkpoint by eval_loss across log history.
    best = None
    for entry in trainer.state.log_history:
        if "eval_loss" in entry:
            if best is None or entry["eval_loss"] < best["eval_loss"]:
                best = entry
    summary = {
        "run_name": args.run_name,
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_epochs": n_epochs,
        "device": "cuda" if use_cuda else "cpu",
        "fp16": fp16,
        "best_epoch": best.get("epoch") if best else None,
        "best_eval_loss": best.get("eval_loss") if best else None,
    }
    with (run_dir / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("[done] " + json.dumps(summary, indent=2))

    # List the surviving checkpoints (best + most recent under save_total_limit=2).
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    finish(
        card,
        metrics={
            "n_train": len(ds_train),
            "n_val": len(ds_val),
            "n_epochs_requested": n_epochs,
            "n_epochs_completed": trainer.state.epoch,
            "best_epoch": summary["best_epoch"],
            "best_eval_loss": summary["best_eval_loss"],
            "device": summary["device"],
            "fp16": summary["fp16"],
            "early_stopped": (trainer.state.epoch is not None
                              and trainer.state.epoch < n_epochs - 1e-3),
        },
        inputs=[args.train],
        outputs=[str(c.relative_to(REPO_ROOT)) for c in ckpts],
        notes=("Stage 3 v2 fine-tune. Recipe: "
               f"lr={cfg['learning_rate']}, wd={cfg['weight_decay']}, "
               f"epochs<= {n_epochs} (early-stop patience "
               f"{cfg.get('early_stopping_patience')}), "
               f"max_in/out={cfg['max_input_length']}/{cfg['max_target_length']}."),
    )


if __name__ == "__main__":
    main()
