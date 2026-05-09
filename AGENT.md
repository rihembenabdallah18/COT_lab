# CoT Knowledge Distillation with ReCEval Evaluation — Implementation Guide (v2)

This file is the execution specification for a coding agent (Claude Code).
It supersedes the v1 plan after the Stage 1–4 diagnostic showed that the
original recipe produced *post-distillation accuracy below the zero-shot
baseline* (4.78% baseline → 2.65% Set A, 2.35% Set B), driven by a combination
of an over-aggressive learning rate, no regularisation, and greedy-decoding
loops in the student.

The companion documents are:

- [doc/RESEARCH.md](doc/RESEARCH.md) — academic motivation and research plan.
- [doc/Related work.md](doc/Related%20work.md) — paper-by-paper notes (Wei, Magister, Wang/SC, ThinkSLM, Ho).
- [doc/Proposed Plan.md](doc/Proposed%20Plan.md) — the user's enhancement plan that v2 implements.
- [doc/Current Notebook.md](doc/Current%20Notebook.md) — the diagnostic that motivated this rewrite.

Read this file end-to-end before starting any stage. Do not skip stages. Stop
at every CHECKPOINT and wait for user approval.

---

## 0. Working mode

**Medium autonomy, stage-gated.** Work one stage at a time in order
(Stages 1–7). At each `CHECKPOINT`: stop, post a short summary of what was
built and what was found, and wait for user approval before proceeding.

If a design choice is unspecified or an assumption breaks (e.g. a dataset
schema is different than described, a library API has changed, training loss
is NaN/flat/increasing, post-distillation accuracy is *worse* than baseline),
**stop and report** — do not silently work around it. The list of
hard-stop conditions is in §6.

After every stage commit to git with a clear message and write a run-card
(see §5 — Observability).

Keep code minimal and readable. This is research code: a few runs, read by a
human grader, not production. Avoid premature abstraction.

---

## 1. What this project tests

The user is reproducing **Magister et al. (ACL 2023)** at small scale on
free-tier hardware (Kaggle/Colab T4, 16 GB), in three weeks. The novel
contribution is a **second evaluation axis on student reasoning**: in
addition to final-answer accuracy, students are scored with **ReCEval**
(Prasad et al., EMNLP 2023) — intra-step, inter-step, and informativeness.

### Research question

> When a small language model is trained via CoT distillation, does
> final-answer accuracy adequately reflect the quality of its generated
> reasoning, or does ReCEval reveal aspects of reasoning quality that
> accuracy alone does not capture?

### Hypotheses

- **H1 — Reproduction.** With the corrected recipe (§3) Set B beats the
  zero-shot baseline on accuracy. (Direction matches Magister; magnitude
  expected to be modest at 220M.)
- **H2 — Filter targets outcome more than process.** The answer-correctness
  filter (Set B) and the calculator-corrected filter (Set C) improve
  accuracy more than they improve ReCEval scores.
- **H3 — Accuracy is incomplete.** A non-trivial fraction of correct-answer
  outputs receive low ReCEval ("right answer, wrong reasoning") and a
  non-trivial fraction of incorrect-answer outputs receive high ReCEval
  ("good reasoning, slip at the end").

### Reference numbers (Ho et al. 2023, FLAN-T5-base on GSM8K)

| Setting | Reported acc. |
|---|---|
| Zero-shot | 2.5% |
| CoT fine-tuning (Magister-style) | 2.96% |
| Direct fine-tuning (Q → A only, no CoT) | 4.93% |

These are the targets v2 is tuned against. The v1 run already cleared zero-shot
(4.78% baseline) but *fell below it* after distillation — the failure
explained in [doc/Current Notebook.md](doc/Current%20Notebook.md).

---

## 2. Constraints

| Constraint | Reason |
|---|---|
| Free-tier T4, 16 GB VRAM | fp16, batch ≤ 8, gradient accumulation, never assume more |
| Kaggle/Colab session timeouts (~12 h Kaggle, variable Colab) | every script must be checkpointed and resumable |
| No paid teacher inference | teacher CoTs come from `itsnamgyu/reasoning-teacher` |
| ReCEval log-likelihood variant only | no pvi training (would require T5-large) |
| Sentence = step (simplified RCU) | SRL is out of scope; permitted by ReCEval paper |
| `seed=42` everywhere | one seed by default; second seed only if time permits |
| Greedy decoding with `no_repeat_ngram_size`/`repetition_penalty`, OR beam=4 | reproducibility; v1 showed pure greedy collapses into loops |

---

## 3. The corrected training & decoding recipe

The v1 recipe (`lr=3e-4`, `weight_decay=0`, 3 epochs, `max_input/target=256`,
pure greedy) caused both students to underperform the baseline. v2 applies
the **aggressive fix** chosen by the user.

### Training (per fine-tuned condition)

| Hyperparameter | v1 value | v2 value |
|---|---|---|
| `learning_rate` | 3e-4 | **5e-5** |
| `weight_decay` | 0.0 | **0.01** |
| `warmup_ratio` | 0.03 | **0.10** |
| `num_epochs` | 3 | **8** (with early stopping on val loss, patience 2) |
| `max_input_length` | 256 | **512** |
| `max_target_length` | 256 | **512** |
| `batch_size` (per device) | 4 | 4 (unchanged) |
| `gradient_accumulation_steps` | 8 | 8 (unchanged; effective batch 32) |
| `fp16` | true | true |
| optimiser | AdamW | AdamW |
| scheduler | linear | linear |
| eval/save strategy | epoch | epoch (best-val checkpoint kept) |

### Decoding (all conditions)

| Hyperparameter | v1 value | v2 value |
|---|---|---|
| `num_beams` | 1 (greedy) | **4** |
| `no_repeat_ngram_size` | 0 | **4** |
| `repetition_penalty` | 1.0 | **1.15** |
| `length_penalty` | 1.0 | 1.0 |
| `max_new_tokens` | 256 | **512** |
| `do_sample` | false | false |

The v1 inference outputs (`outputs/generations/*.jsonl` and `*_repdecoded.jsonl`)
are kept as a v1 archive for the write-up — do not delete.

---

## 4. Conditions matrix

The user has approved **four full conditions** plus a **size-ablation**
slot. StrategyQA is kept as an *optional* extension, not in the main matrix.

| # | Condition | Training data | Format | Purpose |
|---|---|---|---|---|
| 1 | **Baseline** | none (zero-shot) | n/a | reproduction reference (Ho et al. 2.5%) |
| 2 | **Direct FT** | (Q, A) pairs from GSM8K train (no CoT) | input `Q: {q}` → target `{gold_answer}` | reproduces Ho et al.'s 4.93% "fine-tuning" reference |
| 3 | **Set A** | all 7,473 teacher CoTs | input `Q: {q}` → target `{cot} #### {gold_answer}` | no-filter distillation |
| 4 | **Set B** | answer-correctness filter (~3,389) | same as Set A | Magister filter |
| 5 | **Set C** | calculator-corrected filter | same as Set A | **process-aware filter** — rewrite each `A op B = C` in the CoT, then accept iff the *re-parsed* final answer matches gold. Stricter than Set B (rejects "right answer through wrong arithmetic") but adds chains rescued from arithmetic slips. |

**Size ablation (parallel track, optional / time-permitting):**

| 1s–4s | repeat conditions 1, 3, 4 with `google/flan-t5-small` (60M) for the FLOPs/accuracy plot. Skip if Stage 5b runtime is tight. |
|---|---|

**StrategyQA** stays out of the main run. The data prep code lives behind
a `--dataset strategyqa` flag in `src/data/download.py`, but no condition
trains on it unless the user asks.

### How to construct each set

`src/data/filter.py` produces four JSONL files in `data/processed/`:

- `set_a_nofilter.jsonl` — all teacher CoTs (already exists at v1 size 7,473).
- `set_b_magister.jsonl` — `parse_answer(teacher_completion) == gold` (already exists at 3,389).
- `set_c_calculator.jsonl` — for each Set A row, walk the CoT, find equations
  `A op B = C`, replace `C` with the correct value when wrong, then
  re-parse. Keep the row if the calculator-corrected final answer matches
  gold. **Note (post-Stage-2):** this is *not* strictly broader than Set B.
  Empirical sizes: A=7,473, B=3,389, **C=2,635**. Set C rescues ~50 chains
  that fail Magister but whose arithmetic-fix lands on gold; it also evicts
  ~800 chains that pass Magister by coincidence — i.e., the right answer
  appears in trailing prose despite mid-chain arithmetic errors. This makes
  Set C the *process-aware* filter, complementary to Set B's outcome-only
  filter. Print sizes and the B/C contingency table at Stage 2 checkpoint.
- `direct_ft.jsonl` — `{"question": q, "cot": "", "gold_answer": str(g)}`
  with `cot=""`. Reuses the same trainer code; the `{cot} #### {gold}` target
  collapses to ` #### {gold}` (the trainer strips the leading space when
  `cot==""`).

`src/data/parse_answer.py` is the single answer parser used everywhere.
Its existing tests in `tests/test_parse_answer.py` are kept; add tests for
the calculator pass in `tests/test_calculator.py`.

---

## 5. Observability — run-cards and the `status` command

A central pain point in v1 was that every stage had its own ad-hoc status
file. v2 standardises this.

### Run-card convention

Every stage that does work writes one **run-card** under
`outputs/runs/{stage}_{run_name}.json`. The schema:

```json
{
  "stage": "03_train",
  "run_name": "student_set_b",
  "started_at": "2026-05-09T19:30:00Z",
  "completed_at": "2026-05-09T22:14:31Z",
  "duration_seconds": 9871,
  "status": "completed",
  "config_hash": "sha256:...",
  "config_snapshot": { "learning_rate": 5e-5, "...": "..." },
  "inputs": ["data/processed/set_b_magister.jsonl"],
  "outputs": ["outputs/checkpoints/student_set_b/checkpoint-N"],
  "metrics": {
    "n_train": 3050, "n_val": 339,
    "best_epoch": 5, "best_eval_loss": 0.78
  },
  "samples": [{"input": "...", "output": "...", "gold": "..."}],
  "notes": "..."
}
```

A small library `src/utils/runcard.py` provides `start(stage, run_name, config)`
and `finish(card, metrics, outputs, samples=None, notes="")`. Every stage
script imports it.

### `status` command

`python -m src.status` reads every run-card under `outputs/runs/` and prints a
one-screen summary like:

```
Stage  Run                  Status     Duration  Headline
01     download             completed  00:08:22  gsm8k=7473/1319, ho=8792
02     filter               completed  00:00:09  A=7473  B=3389  C=4612  direct=7473
03     student_direct_ft    completed  02:10:11  best_eval_loss=0.42 @ epoch 4
03     student_set_a        completed  04:55:01  best_eval_loss=0.71 @ epoch 6
03     student_set_b        completed  03:12:48  best_eval_loss=0.68 @ epoch 5
03     student_set_c        running    -         epoch 3/8 train_loss=0.81
04     baseline             completed  00:14:38  acc=4.78%
04     student_set_a        completed  00:14:20  acc=tbd
05a    accuracy             pending
05b    receval              pending
06     audit_prep           pending
07     final_analysis       pending
```

Optional: `--json` flag emits the same data as JSON for downstream tooling.

### Per-stage progress logs

In addition to the run-card, long-running stages append a `progress.jsonl`
file inside the stage's run directory. Each line is one event
(`{"t": ..., "event": "epoch_end", "epoch": 2, "train_loss": 0.93, "val_loss": 0.89}`),
which makes it cheap to plot training curves later
(`src/utils/plot_progress.py` produces a PNG per run).

### Where things live

```
outputs/
├── runs/                              # one JSON run-card per stage run
│   ├── 01_download.json
│   ├── 02_filter.json
│   ├── 03_train_student_set_a.json
│   ├── 03_train_student_set_b.json
│   ├── 03_train_student_set_c.json
│   ├── 03_train_student_direct_ft.json
│   ├── 04_inference_baseline.json
│   ├── 04_inference_student_set_*.json
│   ├── 04_inference_student_direct_ft.json
│   ├── 05a_accuracy.json
│   ├── 05b_receval_*.json
│   └── 06_audit_prep.json
├── checkpoints/{run_name}/checkpoint-*    # HF Trainer checkpoints
├── generations/{condition}.jsonl          # v2 outputs
├── generations_v1/{condition}.jsonl       # archived v1 outputs (do not delete)
├── eval_results/                          # accuracy and ReCEval CSVs
├── audit/                                 # blind audit spreadsheet
└── plots/                                 # auto-generated PNGs (loss, length, acc)
```

---

## 6. Repository structure

Build / keep this layout; deviations only with user approval.

```
COT_lab/
├── AGENT.md                              # this file
├── CLAUDE.md                             # short pointer to AGENT.md
├── README.md
├── requirements.txt
├── doc/                                  # research context (read-only)
├── config/
│   └── config.yaml                       # all hyperparameters
├── data/                                 # gitignored
│   ├── raw/{gsm8k, ho_et_al_cots}/
│   └── processed/{set_a,set_b,set_c,direct_ft}.jsonl
├── src/
│   ├── data/
│   │   ├── download.py                   # Stage 1
│   │   ├── filter.py                     # Stage 2 (A, B, C, direct_ft)
│   │   ├── calculator.py                 # NEW: equation rewrite for Set C and acc-w-calc
│   │   └── parse_answer.py
│   ├── train/finetune.py                 # Stage 3 (works for all conditions)
│   ├── inference/generate.py             # Stage 4 (beam + no_repeat_ngram + rep_penalty)
│   ├── eval/
│   │   ├── accuracy.py                   # Stage 5a
│   │   └── receval/{segment, intra_step, inter_step, informativeness, score_chain}.py
│   ├── audit/prepare_audit.py            # Stage 6
│   ├── utils/
│   │   ├── runcard.py                    # NEW: run-card writer
│   │   ├── plot_progress.py              # NEW: PNG plots
│   │   └── flops.py                      # NEW: FLOPs / params accounting
│   └── status.py                         # NEW: `python -m src.status`
├── scripts/
│   ├── 01_download.sh
│   ├── 02_filter.sh
│   ├── 03_train_set_a.sh
│   ├── 03_train_set_b.sh
│   ├── 03_train_set_c.sh
│   ├── 03_train_direct_ft.sh
│   ├── 03_train_smoke.sh
│   ├── 04_inference.sh                   # all five conditions
│   ├── 05a_accuracy.sh
│   ├── 05b_receval.sh
│   ├── 06_audit_prep.sh
│   └── status.sh                         # convenience wrapper
├── notebooks/
│   ├── cot-gpt.ipynb                     # Kaggle orchestration (v1 archived inside)
│   └── final_analysis.ipynb              # Stage 7
├── outputs/                              # gitignored (see §5)
└── tests/
    ├── test_parse_answer.py
    ├── test_calculator.py                # NEW
    └── test_receval_smoke.py
```

---

## 7. Stage-by-stage execution

Each stage ends with a **CHECKPOINT** — post the summary, write the run-card,
commit, and wait for user approval.

### Stage 1 — Environment and data acquisition

**Already done in v1.** Verify only.

- `data/raw/gsm8k/{train,test}.jsonl` exist (7,473 / 1,319).
- `data/raw/ho_et_al_cots/gsm8k_zs_cot_text-davinci-002.json` exists (8,792 records).
- `requirements.txt` resolves on a fresh T4.
- Run-card `01_download.json` written.

**STOP** if any artifact is missing.

### Stage 2 — Build the four training sets

1. Implement `src/data/calculator.py` (an equation rewriter shared with Stage 5a):
   regex `(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)`.
   For each match, compute `lhs op rhs` and replace `result` if wrong.
2. Update `src/data/filter.py` to emit four files: `set_a_nofilter.jsonl`,
   `set_b_magister.jsonl`, `set_c_calculator.jsonl`, `direct_ft.jsonl`.
3. Add `tests/test_calculator.py` (≥ 8 cases including unit-mismatch and
   no-equation chains).
4. Print sizes and report a small contingency table:
   "Set B and Set C agree / disagree" (how many CoTs gain/lose membership
   when we go from answer-only to calculator-corrected).

**Checkpoint:** sizes printed, sample records from each set, contingency
table, run-card `02_filter.json` written.

**Empirical sizes (Stage 2 v2 actuals):** A=7,473, B=3,389, **C=2,635**, direct_ft=7,473.
Set C is *process-aware* and may legitimately be smaller than Set B (it
rejects right-answer-wrong-arithmetic CoTs). The earlier "[Set B, Set A]
band" stop condition was wrong and has been removed. Stop only if Set C
size is empty or > Set A.

### Stage 3 — Fine-tune the four students

1. Update `config/config.yaml` to the v2 recipe (§3).
2. `src/train/finetune.py` already works; add early-stopping
   (`EarlyStoppingCallback`, patience 2 on `eval_loss`).
3. **Smoke run** (`scripts/03_train_smoke.sh`): 200 examples of Set B,
   1 epoch — must finish without error and show decreasing loss.
4. Real runs in this order (cheapest first):
   1. `03_train_direct_ft.sh`  — fastest to converge; sanity check the recipe
   2. `03_train_set_b.sh`      — Magister filter (~3.4K)
   3. `03_train_set_c.sh`      — calculator filter
   4. `03_train_set_a.sh`      — full no-filter set (longest)
5. After each run, write `03_train_{run}.json` run-card with
   `best_eval_loss`, `best_epoch`, `duration_seconds`, and
   `outputs/plots/{run}_loss.png`.

**Mini-checkpoint after Direct FT:** if val loss is *higher* than Magister
reports for similar setups (rough sanity: loss should drop below ~1.0
within 3 epochs), stop — the recipe is still wrong, do not waste a day on
the bigger runs.

**Checkpoint deliverables:**
- 4 checkpoints under `outputs/checkpoints/student_*/`.
- 4 run-cards.
- 4 PNG loss curves under `outputs/plots/`.
- `python -m src.status` shows all four as `completed`.

**Optional size-ablation track** (`flan-t5-small`, conditions 1+3+4):
queue this between Stage 5a and Stage 5b only if budget allows.

### Stage 4 — Inference on the GSM8K test set

1. `src/inference/generate.py` already supports the v2 decoding flags via CLI;
   wire them as defaults from `config.yaml` (`inference.num_beams=4`,
   `inference.no_repeat_ngram_size=4`, `inference.repetition_penalty=1.15`,
   `inference.max_new_tokens=512`).
2. Move the existing v1 generations to `outputs/generations_v1/` (do not
   delete). Then run all five conditions:
   `baseline, student_direct_ft, student_set_a, student_set_b, student_set_c`.
3. Each writes `outputs/generations/{condition}.jsonl` + run-card.
4. Each run-card includes `samples`: 5 (question, generated_cot,
   parsed_answer, gold) tuples for eyeballing.

**Sanity checks:**
- Inspect the first 10 generations of each student; confirm no obvious looping.
- If a condition still produces near-100% looping outputs, stop —
  do not proceed to evaluation.

**Checkpoint:** 5 JSONL files, 5 run-cards, plus a one-line wall-clock
budget report (e.g. "all 5 conditions: 78 minutes, 0.7s/example avg").

### Stage 5 — Evaluation

#### Stage 5a — Accuracy and accuracy-with-calculator

`src/eval/accuracy.py`:

- For each condition, parse the predicted answer and compare to gold.
  Tolerance: `abs(pred - gold) < 1e-6`.
- Accuracy with calculator: feed `generated_cot` through
  `src/data/calculator.py` first, then re-parse and compare.
- Output `outputs/eval_results/accuracy.csv` with columns
  `condition, n, correct, accuracy, correct_w_calc, accuracy_w_calc`.
- Plus `outputs/plots/accuracy_bar.png`.

**Mini-checkpoint:** post the 5×2 table. Compare against Ho et al.'s
2.5 / 2.96 / 4.93 reference numbers. **If any distilled student is below
baseline, STOP — the recipe is still broken; do not proceed to ReCEval.**

#### Stage 5b — ReCEval

This is the project's novel contribution and the largest implementation
piece. Implement carefully and **smoke-test on 20 examples first.**

1. `src/eval/receval/segment.py` — spaCy sentence splitter. Filter empty
   sentences. Cache the loaded `nlp` object.
2. `src/eval/receval/intra_step.py` — DeBERTa-v3 NLI entailment.
   Simplified-RCU: premise = step text, hypothesis = step text. Step score
   = P(entailment). Chain score = min over steps. Document the limitation.
3. `src/eval/receval/inter_step.py` — for step `i`, build prior context =
   `[question, step_1, …, step_{i-1}]`. For each `r` in prior, compute
   P(contradiction) with NLI. Step score = `1 − max_r P_contradiction`.
   Chain score = min over steps. Batch NLI calls (batch 16).
4. `src/eval/receval/informativeness.py` — log-likelihood variant under
   frozen GPT-2 (configurable to Pythia-410M). For each step `i`:
   `log p(gold_answer | q, steps_1..i) − log p(gold_answer | q, steps_1..i-1)`.
   Note: gold answer, not predicted. Chain score = min over steps.
5. `src/eval/receval/score_chain.py` — orchestrates per-chain scoring,
   batches across the test set.
6. Smoke test: 20 examples from Set B output. Hand-inspect 3 chains. Confirm
   probabilities ∈ [0,1] and info gains are sometimes negative (otherwise
   the math is wrong).
7. Full run on all five conditions. Outputs:
   - `outputs/eval_results/{condition}_receval.jsonl` (per-example).
   - `outputs/eval_results/receval_summary.csv` (mean / std / min / max
     per condition × metric).
   - `outputs/plots/receval_violin.png` (per-condition distributions).

**Budget check:** if a single condition exceeds **~3 hours**, drop to a
500-example test subset and clearly note it in the run-card.

**Checkpoint:** accuracy table, ReCEval summary, one fully-worked example
(every step's intra/inter/info plus the chain-level mins), wall-clock per
100 examples, and updated `python -m src.status` output.

### Stage 6 — Manual audit prep (50 examples, blind)

`src/audit/prepare_audit.py`:

- Load `outputs/generations/student_set_b.jsonl`.
- Stratified sample (seed=42): 25 with `parsed == gold`, 25 with
  `parsed != gold`.
- Emit `outputs/audit/audit_blank.csv` with columns
  `id, question, gold_answer, generated_cot, predicted_answer, correct,
  human_label, notes`. **Do not include ReCEval scores** — keep the audit
  blind.
- Emit `outputs/audit/scores_hidden.jsonl` keyed by `id` with all ReCEval
  scalars. The user joins them in *after* labeling.
- Write `outputs/audit/RUBRIC.md` from §6 of `doc/RESEARCH.md`
  (sound / skipped / hallucinated / contradiction / redundant /
  right-answer-wrong-reasoning / wrong-answer-good-until-slip).

**Checkpoint:** three files produced, run-card written.

### Stage 7 — Final analysis

`notebooks/final_analysis.ipynb`:

1. Load `accuracy.csv`, `receval_summary.csv`, all per-example ReCEval
   JSONLs, the filled audit, and `scores_hidden.jsonl`.
2. **Table 1 — Accuracy.** 5 conditions × {acc, acc_w_calc}, with Ho et
   al.'s reference numbers in a side column.
3. **Table 2 — ReCEval.** 5 conditions × 3 metrics (mean ± std).
4. **Table 3 — Cost.** params, FLOPs/inference, training tokens, wall-clock.
   Powered by `src/utils/flops.py`.
5. **Plot 1 — accuracy bar chart** (5 conditions, both accuracy variants).
6. **Plot 2 — ReCEval violin** per condition.
7. **Plot 3 — accuracy vs. FLOPs** (size-ablation overlay if Flan-T5-small
   was run).
8. **Plot 4 — human-vs-ReCEval scatter** (audit only).
9. **Examples section.** 3–4 audit cases: one sound, one each of the
   prominent failure modes, all with CoT + ReCEval scores + human label.
10. Write `outputs/RESULTS.md` with the headline numbers and one-paragraph
    answers to SQ1, SQ2, SQ3.

**Checkpoint deliverables:** notebook runs end-to-end, RESULTS.md produced,
final run-card written.

---

## 8. Hard-stop conditions (do not work around silently)

Stop and report to the user immediately if any of:

- Ho et al.'s JSON schema differs from what `src/data/filter.py` expects.
- A library version fails to install on a fresh T4.
- Training loss is NaN, increasing, or flat at random-baseline level.
- **Post-distillation accuracy is worse than baseline** — same trigger that
  invalidated v1.
- A single ReCEval condition exceeds ~3 hours.
- Disk usage approaching free-tier limits (>40 GB on Kaggle).
- Set C is empty or larger than Set A (the [B, A] band constraint was
  removed after Stage 2 confirmed Set C is process-aware, not broader).
- The recipe in §3 produces no improvement over v1 numbers — escalate
  *before* doing the longer Set A run.

---

## 9. What success looks like

At the end of three weeks the user has:

- 4 trained student checkpoints (Direct FT, Set A, Set B, Set C) on
  Flan-T5-base, optionally a 3-condition repeat on Flan-T5-small.
- 5 sets of generations on the GSM8K test set under v2 decoding.
- An accuracy table with both `acc` and `acc_w_calc`, Ho et al.'s reference
  numbers in the same view.
- ReCEval (intra / inter / info) per condition with mean ± std.
- A 50-example blind audit + cross-tab against ReCEval.
- A final notebook with four tables and four plots, plus `RESULTS.md`.
- Run-cards under `outputs/runs/` such that `python -m src.status` reproduces
  the project state from disk alone.

Good luck. One stage at a time. Stop at every checkpoint.
