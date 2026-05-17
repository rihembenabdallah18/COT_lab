# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

CoT knowledge distillation on GSM8K with FLAN-T5-base. Reproduces Ho et al. (ACL 2023, *Large Language Models Are Reasoning Teachers*) at small scale and adds **ReCEval** (Prasad et al., EMNLP 2023) as a second evaluation axis on student reasoning quality. See [doc/RESEARCH.md](doc/RESEARCH.md) for the full plan and [doc/Related work.md](doc/Related%20work.md) for paper notes.

## Pipeline

Seven stages, each with one script and one src module. Every stage writes a JSON run-card to `outputs/runs/` via `src/utils/runcard.py`; `python -m src.status` reads them and prints a one-screen project state.

| Stage | Script | Module |
|---|---|---|
| 1 | `01_download.sh` | `src/data/download.py` — GSM8K + Ho et al. teacher CoTs |
| 2 | `02_filter.sh` | `src/data/filter.py` + `src/data/calculator.py` — Set A / B / C / Direct FT |
| 3 | `03_train_*.sh` | `src/train/finetune.py` — fine-tune on one set |
| 4 | `04_inference.sh` | `src/inference/generate.py` — beam decoding on the test set |
| 5a | `05a_accuracy.sh` | `src/eval/accuracy.py` — final-answer accuracy |
| 5b | `05b_receval.sh` | `src/eval/receval/` — intra / inter / informativeness |

Conditions: `baseline`, `student_direct_ft`, `student_set_a`, `student_set_b`, `student_set_c`.

All hyperparameters live in `config/config.yaml` (single source of truth). `data/` and `outputs/` are gitignored.

## Constraints

- Free-tier T4 (16 GB). fp16, batch ≤ 8, gradient accumulation.
- Every script must be checkpointed and resumable (Kaggle sessions ~12 h).
- No paid API calls — teacher CoTs come from `itsnamgyu/reasoning-teacher`.
- Seed 42 everywhere.

## Working mode

Stage-gated. Work one stage at a time. If a design choice is unspecified or an assumption breaks (schema differs, loss is NaN, accuracy worse than baseline), **stop and ask** — do not silently work around it. Keep code minimal and readable.