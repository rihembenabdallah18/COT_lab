# CoT Knowledge Distillation with ReCEval Evaluation — Implementation Guide

This README is the execution specification for a coding agent (Claude Code). Read it fully before starting. The companion file `RESEARCH.md` contains the academic motivation; this file contains the build instructions.

---

## 0. How to work on this project

You are operating in **medium autonomy** mode. This means:

- Work through the project **stage by stage** in the order below.
- At the end of each stage there is a **CHECKPOINT** — stop, summarize what you did and what you found, and wait for the user to approve before moving to the next stage.
- Do **not** combine stages. Do **not** skip ahead even if a stage seems trivial.
- If you hit ambiguity inside a stage (a design choice not specified here), ask the user rather than guessing.
- If you discover that an assumption in this README is wrong (e.g. a dataset schema is different than described, a library has changed API), stop and report — don't silently work around it.
- Keep code minimal and readable. This is research code that will be run a few times and read by a human grader, not production software. Avoid premature abstraction.

After every stage you should also commit your work to git with a clear message.

---

## 1. Project goals (so you understand what you're building)

The user is reproducing Magister et al. (ACL 2023, *Teaching Small Language Models to Reason*) at small scale, on free-tier hardware (Kaggle/Colab T4, 16 GB VRAM), in 3 weeks. The added contribution is a **second evaluation metric**: in addition to final-answer accuracy, student outputs are also scored with ReCEval (Prasad et al., EMNLP 2023).

The pipeline trains FLAN-T5-base (250M) on GPT-3-generated CoTs released by Ho et al. 2022, under two filtering conditions:

- **Set A**: no filter (all teacher CoTs)
- **Set B**: answer-correctness filter (Magister's filter — keep only CoTs whose final answer matches the gold answer)

Plus a **non-fine-tuned baseline** (FLAN-T5-base out of the box) as reference.

All three are evaluated on GSM8K test (1,319 examples) using:

1. **Accuracy** (exact-match on the parsed final answer; plus accuracy-with-calculator)
2. **ReCEval** — three sub-metrics: intra-step correctness, inter-step correctness, informativeness

A 50-example **manual audit** of the Set B student's outputs is the final step (this is a human task, not yours; you will only prepare the spreadsheet).

---

## 2. Constraints you must respect

| Constraint | Reason |
|---|---|
| **Hardware: free-tier T4 (16 GB VRAM)** | Use fp16, batch size ≤ 8, gradient accumulation. Never assume more VRAM. |
| **Session timeouts** (~12h Kaggle, variable Colab) | Save checkpoints every epoch. Make every script resumable. |
| **No paid API calls** | All teacher data is pre-released. Do not call OpenAI/Anthropic/Google APIs. |
| **No pvi training for ReCEval** | Use only the NLI variant for correctness and the **log-likelihood variant** for informativeness. Do NOT train T5-large for pvi. |
| **No SRL for RCU extraction** | Use the simplified approximation: each sentence = one step, treat the whole step as both premise and conclusion. The ReCEval reference paper explicitly permits this. |
| **Single random seed (seed=42)** by default | If time permits the user may add a second seed at the end. |
| **Greedy decoding for student inference** | Reproducibility. No sampling unless user requests it. |

---

## 3. Repository structure

Build this layout. Do not deviate without asking.

```
cot-distill-receval/
├── README.md                    # this file (copy in)
├── research_plan.md             # research plan (provided, read-only)
├── requirements.txt             # exact pinned versions
├── config/
│   └── config.yaml              # all hyperparameters in one place
├── data/                        # raw + processed data (gitignored)
│   ├── raw/
│   │   ├── gsm8k/
│   │   └── ho_et_al_cots/
│   └── processed/
│       ├── set_a_nofilter.jsonl
│       └── set_b_magister.jsonl
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py          # Stage 1
│   │   ├── filter.py            # Stage 2 — build Set A and Set B
│   │   └── parse_answer.py      # answer parser, used everywhere
│   ├── train/
│   │   └── finetune.py          # Stage 3 — fine-tune FLAN-T5-base
│   ├── inference/
│   │   └── generate.py          # Stage 4 — generate test outputs
│   ├── eval/
│   │   ├── accuracy.py          # Stage 5a — accuracy + with-calculator
│   │   └── receval/
│   │       ├── __init__.py
│   │       ├── segment.py       # sentence splitting (spaCy)
│   │       ├── intra_step.py    # NLI entailment
│   │       ├── inter_step.py    # NLI contradiction (vs all prior context)
│   │       ├── informativeness.py  # log-likelihood variant
│   │       └── score_chain.py   # aggregate per chain (min over steps)
│   └── audit/
│       └── prepare_audit.py     # Stage 6 — build the audit spreadsheet
├── scripts/
│   ├── 01_download.sh
│   ├── 02_filter.sh
│   ├── 03_train_set_a.sh
│   ├── 03_train_set_b.sh
│   ├── 04_inference.sh
│   ├── 05_eval.sh
│   └── 06_audit_prep.sh
├── notebooks/
│   └── final_analysis.ipynb     # final tables and plots
├── outputs/                     # all generated artifacts (gitignored)
│   ├── checkpoints/
│   ├── generations/
│   ├── eval_results/
│   └── audit/
└── tests/
    ├── test_parse_answer.py
    └── test_receval_smoke.py
```

Use a `.gitignore` to exclude `data/`, `outputs/`, and `__pycache__/`.

---

## 4. Stage-by-stage execution

Each stage ends with a **CHECKPOINT**. At each checkpoint, post a short summary (what was built, what files were produced, any surprises) and wait for user approval.

---

### STAGE 1 — Environment setup and dataset acquisition

**Goal:** get all data on disk, environment working, project skeleton committed.

**Tasks:**

1. Create the repository structure above. Initialize git.
2. Create `requirements.txt` with **pinned versions** of:
   - `torch` (compatible with the CUDA version on the target environment — ask user if unsure)
   - `transformers`
   - `datasets`
   - `accelerate`
   - `sentencepiece`
   - `spacy` plus the `en_core_web_sm` model
   - `pandas`, `numpy`
   - `pyyaml`
   - `tqdm`
3. Write `src/data/download.py`:
   - Downloads GSM8K via `datasets.load_dataset("gsm8k", "main")`. Saves train and test splits as JSONL under `data/raw/gsm8k/`.
   - Downloads Ho et al.'s teacher CoTs from `https://github.com/itsnamgyu/reasoning-teacher`. The release is in `completion_data.tar.gz` on Dropbox / Google Drive (URLs documented in their README). Inspect the GSM8K-related JSON files inside; the schema is *not* fixed in this README — you must inspect and report.
4. Build a tiny smoke test: load 5 GSM8K examples, load 5 teacher CoTs, print them side by side.

**Checkpoint deliverables:**
- Repository skeleton committed
- `requirements.txt` in place
- GSM8K and Ho et al. data on disk
- A short report on Ho et al.'s JSON schema: what fields exist, how to extract `(question, teacher_cot, teacher_predicted_answer)` triples
- The 5-example smoke print

**STOP — wait for user approval before Stage 2.**

---

### STAGE 2 — Build the two filtered training sets

**Goal:** produce `set_a_nofilter.jsonl` and `set_b_magister.jsonl`.

**Tasks:**

1. Implement `src/data/parse_answer.py`. This function must extract the predicted final number from a CoT string. GSM8K's gold answers use the format `#### <number>` — handle that. For free-text CoTs (the teacher's), parse the **last number** in the text as a fallback. Handle commas, decimals, and negative numbers. Return `None` if no number is found.
2. **Write unit tests in `tests/test_parse_answer.py`** with at least 10 examples spanning the formats you'll encounter. Run them.
3. Implement `src/data/filter.py`:
   - For each `(question, teacher_cot)` from Ho et al., write a record `{"question": ..., "cot": teacher_cot, "gold_answer": gsm8k_gold}` to `set_a_nofilter.jsonl`.
   - For Set B, additionally parse the predicted answer from the teacher CoT and **keep the record only if `parsed == gold`**. Save to `set_b_magister.jsonl`.
4. Print Set A size and Set B size. Expected: Set A ≈ 7K, Set B ≈ 5K. If they differ wildly, stop and report.

**Checkpoint deliverables:**
- `set_a_nofilter.jsonl` and `set_b_magister.jsonl` on disk
- Unit tests passing
- Set sizes reported, plus 3 example records from each

**STOP — wait for user approval before Stage 3.**

---

### STAGE 3 — Fine-tune FLAN-T5-base on each set

**Goal:** produce two student checkpoints (`student_set_a/`, `student_set_b/`).

**Tasks:**

1. Write `config/config.yaml` with all hyperparameters in one place:
   - `model_name: google/flan-t5-base`
   - `learning_rate: 3e-4`
   - `batch_size: 4`
   - `gradient_accumulation_steps: 8`  (effective batch 32)
   - `num_epochs: 3`
   - `max_input_length: 256`
   - `max_target_length: 256`
   - `fp16: true`
   - `seed: 42`
   - `eval_every: 1` (per epoch)
   - paths for input data and output checkpoints
2. Write `src/train/finetune.py`:
   - Loads JSONL training data, holds out 10% as validation (deterministic split with `seed`).
   - Input format: `"Q: {question}"` (or whatever Magister-style format works — keep it simple).
   - Target format: `"{cot} #### {gold_answer}"` (matching GSM8K's own format).
   - Uses `transformers.Seq2SeqTrainer` with `predict_with_generate=False` during training.
   - Saves checkpoint per epoch to `outputs/checkpoints/{run_name}/epoch_{N}/`.
   - **Resumable**: detects existing checkpoints and resumes from the latest.
   - Logs train/val loss per epoch to a CSV.
3. Write `scripts/03_train_set_a.sh` and `scripts/03_train_set_b.sh` that invoke the training script with the right `run_name` and input.
4. **Run a 200-example smoke fine-tune first** (subset Set B to 200, 1 epoch, save to a `_smoke` directory) to verify the whole training loop works before committing 5+ hours to a real run.
5. After smoke succeeds, run the **real** Set B training. Then Set A.

**Checkpoint deliverables:**
- Smoke run completed, loss curve looks sane (decreasing)
- `outputs/checkpoints/student_set_a/` and `outputs/checkpoints/student_set_b/` exist
- Best-validation-loss epoch identified for each, recorded in a small JSON
- Final train/val loss reported

**Note for the user:** This stage is the longest (one full day with both runs and buffer). If Kaggle session times out, the resume logic must work. Test resume manually (kill mid-run, restart) on the smoke run.

**STOP — wait for user approval before Stage 4.**

---

### STAGE 4 — Inference on the test set

**Goal:** produce three output JSON files: `baseline.jsonl`, `student_set_a.jsonl`, `student_set_b.jsonl`.

**Tasks:**

1. Write `src/inference/generate.py`:
   - Loads a model (either base FLAN-T5-base, or a fine-tuned checkpoint).
   - Iterates over the GSM8K test set (1,319 examples).
   - **Greedy decoding**, `max_new_tokens=256`, no sampling.
   - Saves each output as a JSONL record: `{"question": ..., "generated_cot": ..., "parsed_answer": ..., "gold_answer": ...}`.
   - Resumable (skip records already present in the output file).
2. Run inference for all three: baseline, Set A student, Set B student.

**Checkpoint deliverables:**
- Three JSONL files in `outputs/generations/`
- Quick sanity check: print 3 examples from each, confirm the format looks right
- Approximate generation wall-clock per condition

**STOP — wait for user approval before Stage 5.**

---

### STAGE 5 — Automatic evaluation

**Goal:** compute accuracy and ReCEval on all three output files.

**Tasks split into 5a (accuracy) and 5b (ReCEval).**

#### Stage 5a — Accuracy

1. `src/eval/accuracy.py` computes:
   - **Exact-match accuracy** on the parsed answer.
   - **Accuracy with calculator**: walk through the generated CoT, find equations of the form `A op B = C`, recompute `A op B` and replace `C` if wrong. Then re-parse the final answer and check. (This is Magister's secondary metric — see their Section 4.1.1 for the worked example. Keep the implementation simple: regex for `\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+`.)
2. Output a results CSV with columns: `condition, n, accuracy, accuracy_w_calc`.

**Mini-checkpoint:** post the accuracy table. Sanity check it against expectations:
- Baseline FLAN-T5-base on GSM8K is typically ~5–15% (no fine-tune)
- After CoT distillation, we expect a meaningful jump (Magister got base→XXL improvements but with much larger models; for FLAN-T5-base expect more modest gains, possibly 15–30%)

If accuracy looks completely wrong (e.g. 0% everywhere, or 100% — both indicate parsing bugs), STOP and debug.

#### Stage 5b — ReCEval

This is the largest implementation piece. Implement carefully.

1. `src/eval/receval/segment.py`:
   - Use spaCy to split a CoT into sentences. Filter out empty / whitespace-only sentences.

2. `src/eval/receval/intra_step.py`:
   - Load NLI model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.
   - For each sentence (= step) in a chain, build (premise, hypothesis) pairs per the simplified-RCU approximation: premise = the sentence itself, hypothesis = the same sentence. **Note:** With one-sentence-as-one-step, intra-step has limited signal — that's expected and a known limitation. Implement it anyway for completeness; document the limitation in a code comment. Score = entailment probability.
   - Step score: P(entailment).
   - Chain score: min over steps.

3. `src/eval/receval/inter_step.py`:
   - For step `i`, build the prior context = [question] ∪ [step_1, ..., step_{i-1}].
   - For each `r` in prior context, compute P(contradiction) with NLI: premise=`r`, hypothesis=`step_i`.
   - Step score: `1 - max_r P_contradiction`.
   - Chain score: min over steps.

4. `src/eval/receval/informativeness.py`:
   - Load a small frozen LM (default: `gpt2`; can swap to `EleutherAI/pythia-410m` if cleaner).
   - For each step `i`, compute:
     - `lp_with = log p(answer | question, steps_1..i)` under the LM
     - `lp_without = log p(answer | question, steps_1..i-1)` under the LM
     - Step score = `lp_with - lp_without`
   - Chain score: min over steps.
   - **Important:** the "answer" here is the *gold* answer, not the predicted answer. Document this.

5. `src/eval/receval/score_chain.py`:
   - For one chain, returns `{"intra": ..., "inter": ..., "info": ...}`.
   - **Batch efficiently.** NLI calls dominate runtime. Process the whole test set with batched NLI calls (batch size 16 typical on T4).

6. Run on all three output files. Save per-example scores to `outputs/eval_results/{condition}_receval.jsonl`. Save summary stats (mean, std, min, max) to a CSV.

7. **Smoke test first**: run on 20 examples from one condition. Hand-inspect 3 chains and their scores. Confirm scores are in plausible ranges (entailment probabilities ∈ [0,1], info gain can be negative, etc.).

**Checkpoint deliverables:**
- `accuracy_results.csv`
- `receval_summary.csv` with per-condition mean/std for each of three metrics
- Per-example ReCEval scores saved
- One worked example output: a single chain with all its step-level and chain-level scores, printed in the report
- Approximate ReCEval wall-clock per 100 examples (for budget tracking)

**STOP — wait for user approval before Stage 6.**

---

### STAGE 6 — Prepare the manual audit

**Goal:** produce a CSV/spreadsheet the user will fill in by hand. **You do not perform the audit.**

**Tasks:**

1. `src/audit/prepare_audit.py`:
   - Load `student_set_b.jsonl` (the Set B / Magister-filter outputs only — see plan).
   - Stratified random sample (seed=42): 25 with correct final answer, 25 with incorrect.
   - For each, output a row with columns:
     - `id` (test set index)
     - `question`
     - `gold_answer`
     - `generated_cot`
     - `predicted_answer`
     - `correct` (bool)
     - `human_label` (empty — user fills in)
     - `notes` (empty)
     - **NOT** included: ReCEval scores. The audit must be blind. Save scores in a separate file `outputs/audit/scores_hidden.jsonl` keyed by `id` — the user joins these in *after* labeling.
2. Save as `outputs/audit/audit_blank.csv`.
3. Write a short `outputs/audit/RUBRIC.md` with the labeling categories from the research plan (sound / skipped step / hallucinated fact / contradiction / redundant / right-answer-wrong-reasoning / wrong-answer-good-reasoning-until-slip).

**Checkpoint deliverables:**
- `audit_blank.csv` (50 rows, blind)
- `scores_hidden.jsonl` (50 entries with ReCEval scores, keyed by id)
- `RUBRIC.md`

**STOP — wait for user approval. The user will do the audit manually.**

---

### STAGE 7 — Final analysis notebook

**Goal:** when the user returns the filled-in `audit_filled.csv`, produce final tables and plots.

**Tasks:**

1. `notebooks/final_analysis.ipynb`:
   - Load `accuracy_results.csv`, `receval_summary.csv`, `audit_filled.csv`, `scores_hidden.jsonl`.
   - **Table 1**: Accuracy and ReCEval (intra/inter/info) per condition (baseline, Set A, Set B). Mean ± std.
   - **Table 2**: Cross-tabulation of human labels vs ReCEval scores. For each label category, report mean ReCEval scores. For each ReCEval metric, report distribution conditional on human-correctness label.
   - **Plot 1**: Per-condition bar chart of accuracy and three ReCEval metrics.
   - **Plot 2**: Scatter of ReCEval score vs. human "is this reasoning sound" judgement (binary).
   - **Examples section**: Pick 3–4 audit examples — one sound, one with each major failure mode — and display CoT + ReCEval scores + human label.

2. Output a final `outputs/RESULTS.md` summarizing the headline numbers in plain markdown.

**Checkpoint deliverables:**
- `final_analysis.ipynb` runs end-to-end
- `RESULTS.md` produced

---

## 5. Things to flag immediately to the user

If you encounter any of the following during execution, stop and report — do not work around them silently:

- Ho et al.'s released JSON schema doesn't match what the README assumes
- Any library API has changed and a pinned version doesn't install
- A training run loss curve looks pathological (NaN, increasing, flat at random-baseline level)
- Accuracy after fine-tuning is *worse* than baseline (almost certainly a data bug)
- ReCEval scoring takes more than ~3 hours per condition (need to subset the test set)
- Disk usage approaching free-tier limits

## 6. What success looks like

At the end of the project the user should have:
- Two trained student checkpoints
- Three sets of generations on the GSM8K test set
- Accuracy + 3 ReCEval scalars per condition
- A filled manual audit
- A final analysis notebook with tables and plots
- Reproducible code that another student could rerun from `requirements.txt` + the scripts in `scripts/`

Good luck. Work one stage at a time. Stop at every checkpoint.