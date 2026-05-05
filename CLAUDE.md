# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3-week NLP research project reproducing Magister et al. (ACL 2023) CoT knowledge distillation at small scale, with ReCEval (Prasad et al., EMNLP 2023) added as a second evaluation axis. The central question: does final-answer accuracy adequately reflect reasoning quality in distilled student models?

**Full execution specification is in [AGENT.md](AGENT.md). Read it before starting any stage.**

## Working Mode

**Medium autonomy, stage-gated.** Work one stage at a time in order (Stages 1–7). At each `CHECKPOINT` in AGENT.md: stop, summarize what was built, and wait for user approval before proceeding. Do not combine or skip stages. If a design choice is unspecified or an assumption breaks, ask rather than guessing.

## Commands

The project is not yet implemented. Commands will be added as stages are completed. When they exist, the entry points are:

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run per-stage scripts in order
bash scripts/01_download.sh
bash scripts/02_filter.sh
bash scripts/03_train_set_b.sh   # Set B first (Magister filter)
bash scripts/03_train_set_a.sh   # Set A (no filter)
bash scripts/04_inference.sh
bash scripts/05_eval.sh
bash scripts/06_audit_prep.sh

# Run tests
python -m pytest tests/
```

## Architecture

Seven sequential pipeline stages, each with a dedicated script and src module:

| Stage | Script | Module | Purpose |
|---|---|---|---|
| 1 | `01_download.sh` | `src/data/download.py` | GSM8K + Ho et al. teacher CoTs → `data/raw/` |
| 2 | `02_filter.sh` | `src/data/filter.py` | Build Set A (all CoTs) and Set B (answer-correct only) → `data/processed/` |
| 3 | `03_train_*.sh` | `src/train/finetune.py` | Fine-tune FLAN-T5-base on each set → `outputs/checkpoints/` |
| 4 | `04_inference.sh` | `src/inference/generate.py` | Greedy decoding on GSM8K test set (1,319 examples) → `outputs/generations/` |
| 5 | `05_eval.sh` | `src/eval/accuracy.py` + `src/eval/receval/` | Accuracy + ReCEval scoring → `outputs/eval_results/` |
| 6 | `06_audit_prep.sh` | `src/audit/prepare_audit.py` | 50-example blind audit spreadsheet → `outputs/audit/` |
| 7 | (notebook) | `notebooks/final_analysis.ipynb` | Tables, plots, final results |

All hyperparameters live in `config/config.yaml` (single source of truth). `data/` and `outputs/` are gitignored.

## ReCEval Scoring (Stage 5b)

The core novel component lives in `src/eval/receval/`. It computes three per-chain scalars, each aggregated as the **minimum over steps**:

- **Intra-step** (`intra_step.py`): NLI entailment probability using `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. Simplified-RCU approximation: each sentence = one step, treated as both premise and conclusion.
- **Inter-step** (`inter_step.py`): `1 - max P(contradiction)` between step `i` and all prior context (question + steps 1..i-1), using the same NLI model.
- **Informativeness** (`informativeness.py`): log-likelihood gain `log p(gold_answer | question, steps_1..i) - log p(gold_answer | question, steps_1..i-1)` under a frozen GPT-2 or Pythia-410M. Uses **gold answer**, not predicted answer.

Sentence splitting uses spaCy (`segment.py`). NLI calls must be batched (batch size ~16) to fit T4 VRAM budget.

## Key Constraints

- **Hardware**: free-tier T4 (16 GB VRAM). Always use fp16, batch ≤ 8, gradient accumulation.
- **Resumability**: every training and inference script must checkpoint and skip already-completed work on restart.
- **No API calls**: all teacher data comes from Ho et al.'s pre-released CoTs (`itsnamgyu/reasoning-teacher`).
- **No pvi informativeness**: use log-likelihood variant only (no T5-large training).
- **No SRL**: sentence = step approximation throughout.
- **Greedy decoding** for all inference (reproducibility).
- **Seed 42** for all random operations.

## Stop Conditions

Stop and report to the user (do not work around) if:
- Ho et al.'s JSON schema doesn't match expectations
- A library version fails to install
- Training loss is NaN, increasing, or flat at random-baseline level
- Post-distillation accuracy is *worse* than baseline
- ReCEval scoring exceeds ~3 hours per condition
- Disk usage approaches free-tier limits

## Answer Parsing

`src/data/parse_answer.py` is used everywhere. GSM8K gold format: `#### <number>`. Free-text fallback: last number in the string. Must handle commas, decimals, negatives; return `None` if nothing found. Has ≥10 unit tests in `tests/test_parse_answer.py`.
