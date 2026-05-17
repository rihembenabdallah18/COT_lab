# COT_lab

Small-scale CoT knowledge distillation on GSM8K with FLAN-T5-base, evaluated jointly with final-answer accuracy and ReCEval reasoning-quality scores.

Reproduces Ho et al. (ACL 2023, *Large Language Models Are Reasoning Teachers*); adds Prasad et al. (EMNLP 2023, *ReCEval*) as a reasoning-quality axis.

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

Each stage is one script. All hyperparameters are in [config/config.yaml](config/config.yaml). Data and outputs are gitignored.

```bash
bash scripts/01_download.sh        # GSM8K + Ho et al. teacher CoTs
bash scripts/02_filter.sh          # Build Set A / B / C / Direct FT
bash scripts/03_train_direct_ft.sh # Fine-tune (one script per condition)
bash scripts/03_train_set_b.sh
bash scripts/03_train_set_c.sh
bash scripts/03_train_set_a.sh
bash scripts/04_inference.sh       # All 5 conditions on the GSM8K test set
bash scripts/05a_accuracy.sh       # Final-answer accuracy
bash scripts/05b_receval.sh        # Intra / inter / informativeness

python -m src.status               # Project-wide state from outputs/runs/*.json
pytest tests/                      # Unit tests
```

## Notebooks

- [notebooks/kaggle.ipynb](notebooks/kaggle.ipynb) — full pipeline orchestrator for Kaggle (T4).
- [notebooks/results.ipynb](notebooks/results.ipynb) — load `outputs/eval_results/` and render the final tables and plots.

## Layout

```
config/   single source of truth for hyperparameters
src/      data, train, inference, eval, utils
scripts/  one shell wrapper per stage
tests/    pytest unit tests
doc/      research plan and related work
```