# Pipeline Recommendations — Post Stages 1–5b

_Based on the findings in `notebooks/results.ipynb`. Written 2026-05-13._

---

## 1. Situation summary

Five conditions were evaluated: `baseline`, `student_direct_ft`, `student_set_a/b/c`.
Three core problems emerge from the results:

| Problem | Evidence |
|---|---|
| All CoT students fall below the un-finetuned baseline | Set A 2.50%, B 3.03%, C 3.18% vs. baseline 4.32% |
| Eval-time calculator rewriting is a complete no-op | `accuracy_w_calc == accuracy` for every condition |
| ReCEval cannot distinguish the three CoT conditions | Δintra < 0.01, Δinter < 0.02, Δinfo < 0.08 — all within noise |

The two root causes driving all three problems are:
1. **Teacher data quality**: Ho et al.'s `text-davinci-002` zero-shot chains are correct only ~45% of the time, and most failures are *premise* errors (wrong equation setup), not arithmetic errors. Set B and Set C both try to filter these out, but premise-error chains that arrive at the right final answer by luck survive Set B, and even Set C only fixes arithmetic slips — not logical setup errors.
2. **Model capacity**: FLAN-T5-base (250M params) learning multi-step arithmetic CoTs from a noisy 2.5K–7.5K example dataset. The model cannot generalise the reasoning; it memorises equation templates, which collapses to paraphrasing the question with wrong numbers.

---

## 2. High-impact changes (fix the accuracy drop)

### 2.1 Add online calculator decoding at inference time

**What**: during beam search, after the model emits an equation token (`A op B = C`), run the real calculator to override the RHS before sampling the next token (Magister et al. 2022 §3).

**Why**: the notebook confirms eval-time rewriting does nothing because arithmetic errors in the *final answer position* were already propagated and committed. Online decoding intervenes *before* the model locks in the wrong answer.

**Expected gain**: Magister et al. report +1.97 pp for FLAN-T5-base (2.96% → 4.93%) with this one change — more than the gap between our best student and baseline.

**Implementation**: modify `src/inference/generate.py` to pass a custom `LogitsProcessor` or post-process inside a generation loop after each `=` token in the beam; update `scripts/04_inference.sh` to expose a `--calculator` flag.

**Cost**: inference-only change, no retraining needed. Should run inside the existing T4 budget.

---

### 2.2 Add a stricter "premise-correct" filter (Set D)

**What**: after Set C's arithmetic rewrite, also check that every equation's *inputs* appear in the problem text or are derivable from a prior equation output. Discard chains where a number appears in an equation but has no textual provenance (i.e., the model hallucinated an operand).

**Why**: the qualitative examples in Section 6 of the notebook show that all three CoT students make premise errors ("she needs to answer 70 + 60% = 80"). Set C corrects `5 * 4 = 18 → 20` but does not reject a chain that says `70 + 60% = 80` when the correct operation is `70 * 0.60 = 42`. A provenance check on inputs would filter these.

**Expected gain**: unknown, but Set C already shows the accuracy trend (C > B > A) continues as the filter becomes stricter. A Set D with ~1K–2K highly-clean chains may outperform Set C at 2.6K despite the size reduction — this is worth one extra ablation condition.

**Implementation**: extend `src/data/filter.py`; add `scripts/03_train_set_d.sh`. The audit spreadsheet from Stage 6 is a good place to manually validate a 50-example sample of the Set D chains before running full training.

---

### 2.3 Answer-weighted loss during training

**What**: multiply the per-token cross-entropy loss by a higher weight (e.g. ×3–5) for tokens in the `#### N` suffix relative to the reasoning tokens.

**Why**: the notebook shows Direct FT (Q → `#### N` only) reaches 5.00% — the model already has a strong answer prior. CoT distillation must suppress that prior and reroute through a much longer, noisier chain, which degrades accuracy. Up-weighting the answer tokens re-anchors the gradient signal at the correct answer without dropping the CoT supervision entirely.

**Expected gain**: should close most of the CoT vs. Direct FT gap. Cheap to implement (one `loss_mask` tensor change in `src/train/finetune.py`).

**Implementation**: in the training loop, build a mask that identifies `####`-suffix tokens; multiply `loss_unreduced[mask] *= answer_weight` before the mean. Expose `answer_weight` in `config/config.yaml` with a default of 1.0 to keep existing behaviour.

---

## 3. Medium-impact changes (fix ReCEval)

### 3.1 Fix the Direct FT ReCEval artifact

**What**: flag / exclude Direct FT from the ReCEval comparison table. Its high inter-step (0.723) and positive informativeness (0.691) are driven by the single-step `#### N` output being out-of-distribution for the scorer — it trivially satisfies inter-step coherence because there is only one step.

**Why**: including Direct FT in the ReCEval scatter plot actively misleads the correlation analysis. The notebook text already notes this; the table and plots should visually mark it differently (hatched bar, asterisk, footnote).

**Implementation**: notebook change only — grey out the Direct FT bars in the ReCEval panels, add a footnote, and add a separate panel or table for the four multi-step conditions only. No pipeline change needed.

---

### 3.2 Add per-problem-type ReCEval breakdown

**What**: split the 1,319 test problems by difficulty tier (1-step, 2–3 steps, 4+ steps — approximated by gold chain length in the reference dataset) and report ReCEval separately per tier.

**Why**: all three CoT students are currently averaging over easy and hard problems together. The very low inter-step (0.063–0.073) and negative informativeness may be dominated by 1-step problems where the student emits a trivial paraphrase. On harder, genuinely multi-step problems, Set C's stricter supervision may produce measurably better chains.

**Implementation**: add a difficulty column to the generations JSONL (derivable from the GSM8K `answer` field length or step count), then group the ReCEval records by tier in the notebook. No pipeline changes.

---

### 3.3 Reconsider the informativeness scorer baseline

**What**: the notebook reports negative informativeness (−2.35 to −2.45) for all CoT students, using a frozen GPT-2 log-likelihood scorer. GPT-2 has very different tokenization and vocabulary priors from FLAN-T5, meaning the baseline `log p(answer | question)` is already low, so the gain from adding steps may not register. Try Pythia-410M (also supported in the spec) or compute the metric relative to the baseline condition rather than relative to the GPT-2 prior.

**Why**: if the negative informativeness is a scorer artifact rather than a genuine signal, it undermines the ReCEval story. A relative informativeness metric (Δinfo = CoT student info − baseline info) would be more interpretable.

**Implementation**: `src/eval/receval/informativeness.py` — add the `pythia-410m` backend (already listed in the spec as allowed) and re-run Stage 5b for the CoT conditions. Alternatively, compute relative-to-baseline in the notebook post-hoc using the existing JSONL files.

---

## 4. Low-cost / narrative changes (strengthen the write-up)

### 4.1 Reframe the comparison table

The current summary table implies Direct FT is a ceiling for CoT students. Add a column: **"reasons well?"** (binary judgement: produces multi-step output with `####` and inter-step > 0.1). This makes the decoupling argument explicit in the table rather than buried in Section 9.

### 4.2 Report accuracy on easy-only subset

Filter the test set to problems that have a 1-line gold solution (single arithmetic operation). CoT students should perform at least as well as Direct FT on these, because the premise is trivial. If they don't, it means the distillation is hurting even the simplest cases — a strong signal for model-capacity diagnosis.

### 4.3 Run a single extra epoch with unfrozen encoder

In the v2 recipe the encoder is presumably unfrozen (FLAN-T5 full fine-tune). If the encoder was frozen, unfreeze it — the model needs to remap its internal number representations to align with the CoT format.

### 4.4 Add confidence intervals to the accuracy table

With n=1,319 and the best CoT accuracy at 3.18%, the 95% CI is ≈ ±0.94 pp (binomial). Set B (3.03%) and Set C (3.18%) are not statistically distinguishable. The paper should report CIs or a McNemar test rather than treating the 0.15 pp gap as meaningful.

---

## 6. v4 training recipe (planned changes)

The following changes are planned to push all CoT students above the baseline (4.32%) and above Direct FT (5.00%) at least with calculator correction. Documented here so individual changes can be reverted by restoring the old value.

### config/config.yaml

| Parameter | v2 value | v4 value | Reason |
|---|---|---|---|
| `num_epochs` | 8 | 12 | All CoT students had `best_epoch ≈ 8` — still improving at cutoff |
| `weight_decay` | 0.01 | 0.05 | More regularisation for small CoT sets (set_b: 3K, set_c: 2.4K) |

### src/train/finetune.py (inside `Seq2SeqTrainingArguments`)

| Parameter | v2 value | v4 value | Reason |
|---|---|---|---|
| `label_smoothing_factor` | not set (0.0) | 0.1 | CoT chains are noisy labels; smoothing improves generalisation |
| `lr_scheduler_type` | not set (linear) | `"cosine"` | Smoother LR decay over 12 epochs avoids sharp cutoff |

---

## 5. Prioritised action list

| Priority | Action | Stage affected | Effort | Expected payoff |
|---|---|---|---|---|
| 1 | Online calculator decoding | Stage 4 (inference) | Low — no retraining | +~2 pp accuracy, free |
| 2 | Answer-weighted loss | Stage 3 (training) | Low — single tensor change | Narrows CoT vs. Direct FT gap |
| 3 | Fix Direct FT ReCEval artifact in notebook | Stage 7 (notebook) | Trivial | Cleans up misleading scatter |
| 4 | Set D filter (premise provenance check) | Stages 2–3 | Medium — new filter logic + retrain | Tests the accuracy-filter trend further |
| 5 | Per-tier ReCEval breakdown | Stage 7 (notebook) | Low — notebook only | May rescue the ReCEval differentiation story |
| 6 | Pythia-410M informativeness rescore | Stage 5b | Medium — re-run scorer | Validates or invalidates the negative info finding |
| 7 | Add accuracy CIs / McNemar test | Stage 7 (notebook) | Trivial | Research integrity |
| 8 | Reframe Direct FT in comparison table | Stage 7 (notebook) | Trivial | Clearer narrative |

**Recommended minimal path** (if time is limited): actions 1, 3, 7, and 8 require no retraining and together fix the two biggest problems — the accuracy drop (action 1) and the misleading ReCEval comparison (action 3). They could be done in a single session.
