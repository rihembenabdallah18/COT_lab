# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 3-week NLP research project reproducing Magister et al. (ACL 2023) CoT knowledge distillation at small scale, with ReCEval (Prasad et al., EMNLP 2023) added as a second evaluation axis. The central question: does final-answer accuracy adequately reflect reasoning quality in distilled student models?

**Full execution specification is in [AGENT.md](AGENT.md) (currently v2 — supersedes v1 after the diagnostic in [doc/Current Notebook.md](doc/Current%20Notebook.md)). Read AGENT.md before starting any stage.**

Supporting context is in `doc/`:
- `RESEARCH.md` — research plan
- `Related work.md` — paper-by-paper notes (Wei, Magister, Wang/SC, ThinkSLM, Ho)
- `Proposed Plan.md` — the user's enhancement plan that AGENT.md v2 implements
- `Current Notebook.md` — the diagnostic that motivated the v2 rewrite

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
bash scripts/03_train_direct_ft.sh  # Direct FT first (cheapest sanity check)
bash scripts/03_train_set_b.sh      # Magister filter (~3.4K)
bash scripts/03_train_set_c.sh      # Calculator-corrected filter
bash scripts/03_train_set_a.sh      # No filter (full ~7.5K, longest)
bash scripts/04_inference.sh
bash scripts/05a_accuracy.sh
bash scripts/05b_receval.sh

bash scripts/06_audit_prep.sh

# Project-wide status across all stages (reads outputs/runs/*.json):
bash scripts/status.sh

# Run tests
python -m pytest tests/
```

## Architecture

Seven sequential pipeline stages, each with a dedicated script and src module:

| Stage | Script | Module | Purpose |
|---|---|---|---|
| 1 | `01_download.sh` | `src/data/download.py` | GSM8K + Ho et al. teacher CoTs → `data/raw/` |
| 2 | `02_filter.sh` | `src/data/filter.py` + `src/data/calculator.py` | Build Set A (no filter), Set B (answer-correct), Set C (calculator-corrected), Direct FT (Q→A only) → `data/processed/` |
| 3 | `03_train_*.sh` | `src/train/finetune.py` | Fine-tune FLAN-T5-base on each training set → `outputs/checkpoints/` |
| 4 | `04_inference.sh` | `src/inference/generate.py` | Beam=4 + no_repeat_ngram=4 + repetition_penalty=1.15 on GSM8K test (1,319) → `outputs/generations/` |
| 5a | `05a_accuracy.sh` | `src/eval/accuracy.py` | Accuracy + accuracy-with-calculator → `outputs/eval_results/accuracy.csv` |
| 5b | `05b_receval.sh` | `src/eval/receval/` | ReCEval (intra/inter/info) → `outputs/eval_results/receval_summary.csv` |
| 6 | `06_audit_prep.sh` | `src/audit/prepare_audit.py` | 50-example blind audit spreadsheet → `outputs/audit/` |
| 7 | (notebook) | `notebooks/final_analysis.ipynb` | Tables, plots, FLOPs/cost, final results |

Cross-cutting: every stage writes a JSON run-card to `outputs/runs/{stage}_{run}.json` via `src/utils/runcard.py`. `python -m src.status` reconstructs project state from those run-cards.

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
- **Beam=4 + no_repeat_ngram=4 + repetition_penalty=1.15** for all inference (reproducibility; pure greedy collapsed into loops in v1).
- **Seed 42** for all random operations.
- **v2 training recipe**: `lr=5e-5`, `weight_decay=0.01`, `epochs=8` (early-stop patience 2), `max_input/target=512`, `warmup_ratio=0.10`. The v1 recipe (lr=3e-4, no decay, 3 epochs, max_len=256) caused post-distillation accuracy *below baseline*.

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
