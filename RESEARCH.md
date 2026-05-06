# Research Plan — Chain-of-Thought Knowledge Distillation from LLMs to SLMs with Reasoning-Quality Evaluation

**Author:** [Your Name]
**Course / Supervisor:** [To fill in]
**Duration:** 3 weeks (~75 hours total, ~5 h/day)
**Date:** May 2026

---

## 1. Summary

This project reproduces, at small scale and with limited compute, the chain-of-thought (CoT) knowledge distillation pipeline of Magister et al. (ACL 2023, *Teaching Small Language Models to Reason*). Beyond simple reproduction, it adds a **second evaluation axis**: in addition to final-answer accuracy, the student model's generated reasoning chains are scored using ReCEval (Prasad et al., EMNLP 2023), a reference-free framework for measuring reasoning correctness and informativeness. A focused **filter ablation** (no filter vs. answer-correctness filter) and a **manual audit** of 50 outputs make the dual-metric evaluation defensible.

The central claim being tested is whether final-answer accuracy alone — the standard metric in CoT distillation work — adequately reflects whether reasoning capability has actually transferred to the student.

---

## 2. Background and Motivation

CoT prompting reliably elicits step-by-step reasoning in large language models (≥ tens of billions of parameters), but small models (< 10B) typically produce incoherent CoTs and may even lose accuracy when prompted to reason explicitly. Magister et al. (2023) close this gap via knowledge distillation: a large teacher (e.g. PaLM 540B, GPT-3 175B) is prompted to generate CoTs on an existing supervised dataset; CoTs whose final answer is incorrect are filtered out; a small student (T5 family) is fine-tuned on the surviving (question → CoT + answer) pairs. They report substantial accuracy gains, e.g. on GSM8K, T5-XXL improves from 8.11% to 21.99%.

A natural question follows. Magister et al.'s filter accepts any CoT whose **final number** is correct, regardless of whether the reasoning chain is internally consistent, free of hallucinations, or step-wise informative. As a result:

- Some accepted training examples reach the right answer through flawed reasoning.
- The student's reported gain is measured purely on final-answer accuracy — leaving open whether reasoning quality genuinely improved or whether the student learned shortcut patterns.

ReCEval (Prasad et al., 2023) provides a reference-free way to score reasoning chains along two complementary axes: **correctness** (intra-step entailment, inter-step non-contradiction) and **informativeness** (information gain toward the final answer), all computed with off-the-shelf NLI and LM components. Applying ReCEval to the *student's own outputs* — not just to teacher data — directly measures whether reasoning capability has transferred, complementing accuracy.

This project combines the two: a small-scale Magister-style distillation pipeline, evaluated jointly with accuracy and ReCEval, on a single student and a single dataset, with a filter ablation to give the comparison something to say.

---

## 3. Research Question and Hypotheses

### Research question

> When a small language model is trained via CoT knowledge distillation, does final-answer accuracy adequately reflect the quality of its generated reasoning chains, or does ReCEval-based evaluation reveal aspects of reasoning quality that accuracy alone does not capture?

### Sub-questions

- **SQ1 (Reproduction).** Does a Magister-style filtered distillation, applied at small scale (FLAN-T5-base, free-tier GPU), reproduce the qualitative finding that distillation improves student final-answer accuracy on GSM8K relative to a non-fine-tuned baseline?
- **SQ2 (Filter effect on reasoning quality).** Does the answer-correctness filter improve only accuracy, or does it also improve ReCEval-measured reasoning quality of the student?
- **SQ3 (Metric agreement).** On a manually labeled subset of student outputs, do ReCEval scores correlate with human judgement of reasoning quality? Where do they diverge?

### Hypotheses

- **H1.** Distillation improves accuracy relative to the non-fine-tuned baseline (expected; reproduction).
- **H2.** The answer-correctness filter improves accuracy more than it improves ReCEval scores — i.e. the filter is well-targeted at outcomes but only weakly targeted at process.
- **H3 (null acceptable).** A non-trivial fraction of correct-answer outputs receive low ReCEval scores ("right answer, flawed reasoning"), and a non-trivial fraction of incorrect-answer outputs receive high ReCEval scores ("good reasoning, slip at the end"). If true, this directly supports the claim that accuracy alone is an incomplete metric.

A null result for H2 (filter helps both equally) is also reportable and informative — it would suggest that for this particular failure mode, accuracy is in fact a reasonable proxy.

---

## 4. Scope

### In scope

- **Single dataset:** GSM8K (grade-school math word problems). Chosen because it is the headline result in Magister et al., has a clean automatic correctness check, and has multi-step reasoning that makes ReCEval meaningful.
- **Single student model:** FLAN-T5-base (250M parameters). Chosen for direct architectural comparability to Magister et al., free-tier feasibility, and fast iteration.
- **Two training conditions:** (a) no filter on teacher CoTs; (b) answer-correctness filter (Magister's filter). Plus a non-fine-tuned baseline reference.
- **Pre-released teacher CoTs:** GPT-3-generated CoTs on GSM8K released by Ho et al. 2022 (`itsnamgyu/reasoning-teacher` GitHub repo). No teacher inference is performed.
- **Two evaluation metrics:** final-answer accuracy and ReCEval (intra-step, inter-step, informativeness).
- **One manual audit:** 50 student outputs, blind-labeled by the author.

### Out of scope (and why)

| Excluded | Reason |
|---|---|
| Larger students (T5-XL, T5-XXL, Qwen-3B+) | Will not fit on free-tier T4 with full fine-tuning |
| ReCEval pvi-based informativeness | Requires training T5-large twice; uses too much budget |
| Running the teacher model directly | API cost, reproducibility; pre-released CoTs avoid this |
| Multi-task / multi-dataset setup | Doubles work, dilutes the central claim |
| ReCEval as a *filter* (not just metric) | More ambitious variant; deferred to future work |
| Self-consistency, decomposition, RL signals | Distinct research questions |

### Hardware and software

- Free-tier Kaggle / Google Colab notebooks (NVIDIA T4, 16 GB VRAM, ~12-hour sessions)
- HuggingFace Transformers, Datasets, Accelerate
- DeBERTa-v3-large NLI (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) for ReCEval correctness components
- GPT-2 or Pythia-410M as the frozen LM for the log-likelihood informativeness variant
- spaCy for sentence segmentation
- Checkpoints and intermediate artifacts mirrored to a private HuggingFace Hub repository for cross-session persistence

---

## 5. Pipeline Description

The full pipeline runs in seven sequential stages.

### Stage 1 — Data preparation

Download three resources, all from HuggingFace or GitHub, no API calls required:

1. **GSM8K** from `openai/gsm8k` (HuggingFace). Provides ~7.5K train and 1,319 test problems with gold answers.
2. **Teacher CoTs** from Ho et al. 2022 (`itsnamgyu/reasoning-teacher`). Provides GPT-3-generated CoTs for GSM8K training set in JSON format.
3. **Pretrained student weights** for FLAN-T5-base from HuggingFace.

Construct a unified training set of triples `(question, teacher_cot, gold_answer)` for each training example.

### Stage 2 — Filter construction

Build two training sets from the same source data:

- **Set A — No filter.** All teacher CoTs retained (~7K examples).
- **Set B — Answer-correctness filter (Magister).** Parse the final number from each teacher CoT (last numeric expression after `####` or in the final sentence), compare to gold; keep only matches. Expected size: ~5K–5.5K.

Record dataset sizes precisely. Both sets share the same training split — only the filter differs.

### Stage 3 — Student fine-tuning

Fine-tune FLAN-T5-base independently on each of Set A and Set B, producing two student checkpoints. Identical hyperparameters across both runs to isolate the filter's effect:

- Input format: question only
- Target format: CoT followed by answer (Magister's format)
- Optimizer: AdamW, learning rate ~3e-4, linear warmup
- Batch size 4–8 with gradient accumulation to effective batch ~32
- Mixed precision (fp16) to fit comfortably in 16 GB
- 3 epochs with early stopping on a held-out 10% validation slice
- Checkpoint per epoch; select best-validation-loss checkpoint

Expected wall-clock: 3–5 hours per run on a T4. Both runs together: one full day with buffer.

### Stage 4 — Inference on the test set

Run each fine-tuned student plus the non-fine-tuned baseline on GSM8K's 1,319-example test set. Generation settings: greedy decoding (deterministic for reproducibility), max length 256 tokens. Save all outputs to disk as JSON: `(question, generated_cot, parsed_answer, gold_answer)`.

This yields three output files: `baseline.json`, `nofilter.json`, `magister.json`.

### Stage 5 — Automatic evaluation

#### 5.1 Accuracy

For each of the three output files, parse the predicted answer from each CoT and compare to gold. Report:

- Overall accuracy
- Accuracy with an external calculator (Magister's secondary metric — recompute the right-hand side of any equations the model produces, replacing arithmetic mistakes; isolates reasoning errors from arithmetic errors)

#### 5.2 ReCEval scoring

For each generated CoT, compute three scalars:

- **Intra-step correctness.** Split the CoT into sentences (spaCy). Following the simplified-RCU approximation explicitly permitted by the ReCEval reference (treating the full step as both premise and conclusion), for each step compute the entailment probability between the prior context within the step and the step's conclusion using a DeBERTa-v3 NLI model. Take the minimum across steps.
- **Inter-step correctness.** For each step, compute the maximum contradiction probability between the step's conclusion and (i) the question, (ii) every prior step. Step-level score = 1 − max contradiction. Chain-level = minimum across steps. Per ReCEval's empirical finding, the full prior context (not only the immediately preceding step) is used.
- **Informativeness (log-likelihood variant).** For each step, compute `log p(answer | question, prior_steps, current_step) − log p(answer | question, prior_steps)` under a small frozen LM (GPT-2 or Pythia-410M). Take the minimum across steps. Positive = informative; negative = unhelpful or misleading.

The pvi-based variant (which requires training T5-large) is **not** used; the log-likelihood variant requires no training and runs on a T4.

Aggregate: report mean and standard deviation of each metric across the test set, per condition.

### Stage 6 — Manual audit (50 examples)

Randomly sample 50 student outputs **from the Magister-filter condition only** (the most representative system). Stratify: 25 with correct final answer, 25 with incorrect final answer.

**Procedure (blind labeling):**

1. Build a labeling rubric *before* looking at any ReCEval scores. Categories (initial draft):
   - Sound reasoning (every step follows, contributes)
   - Skipped step (logical jump, conclusion not derived)
   - Hallucinated fact (information not in question)
   - Internal contradiction
   - Redundant / off-topic
   - Right answer, wrong reasoning (final number correct but chain doesn't justify it)
   - Wrong answer, sound reasoning until a slip

   Refine after labeling the first 5–10 examples; re-label any earlier examples affected by rubric changes.

2. Open each generated CoT in a spreadsheet. Read it. Assign one or more category labels. Do **not** look at ReCEval scores during this step.

3. Once all 50 are labeled, join with ReCEval scores and cross-tabulate.

**Outputs of the audit:**

- An agreement table: which ReCEval sub-metric (intra / inter / info) is low when which human-labeled failure mode appears?
- An overall agreement rate (e.g. "ReCEval intra-step correctness flagged X of Y skipped-step cases").
- 3–4 illustrative examples for the write-up, including any prominent disagreements.

Time estimate: 3–5 minutes per example × 50 = 3–4 hours of focused work, plus 1–2 hours for analysis and tabulation.

### Stage 7 — Analysis and write-up

Produce, in this order:

1. A reproduction-level result: baseline vs. distilled accuracy, addressing SQ1.
2. Filter-effect tables: the two distilled conditions side by side on accuracy and on each of the three ReCEval metrics, addressing SQ2.
3. Manual-audit results: agreement table plus qualitative examples, addressing SQ3.
4. A discussion of where accuracy and ReCEval agree vs. disagree — the central claim of the project.
5. Limitations: small student, single dataset, simplified RCU approximation, free-tier compute, no pvi variant, single random seed (or two if time permits).

---

## 6. Experimental Matrix

| Condition | Training data | Training run? | Evaluated on | Purpose |
|---|---|---|---|---|
| **Baseline** | none (zero-shot) | no | full test set | reproduction reference |
| **No-filter distillation** | Set A (all teacher CoTs) | yes | full test set + audit subset | filter ablation lower bound |
| **Magister-filter distillation** | Set B (answer-correct only) | yes | full test set + audit subset (manual audit drawn from this) | reproduction of Magister + main result |

**Total training runs: 2.** Total inference passes: 3. Total ReCEval scoring passes: 3. Manual audit: 1 (on the Magister-filter outputs).

---

## 7. Timeline (3 weeks, ~75 hours)

### Week 1 — Setup, baselines, and pipeline shakedown (~25 h)

- **Days 1–2 (~10 h).** Re-read Magister et al. and ReCEval reference carefully. Set up the Kaggle/Colab environment. Install dependencies. Push a "hello world" notebook to the project repo.
- **Days 3–4 (~10 h).** Download GSM8K, Ho et al.'s teacher CoTs, and FLAN-T5-base. Build Set A and Set B. Verify dataset sizes match expectations. Run a 200-example end-to-end shakedown: tiny fine-tune, inference, accuracy parse, one ReCEval call. **Goal: every component runs on real data, end-to-end, before any full run.**
- **Day 5 (~5 h).** Run baseline inference (no fine-tuning) on the full test set. Compute baseline accuracy. Implement and validate the answer-parsing function on these outputs.

### Week 2 — Full training and evaluation (~30 h)

- **Days 6–7 (~10 h).** First full fine-tuning run (Set B / Magister filter). Monitor validation loss. Save checkpoints to HuggingFace Hub. **Buffer for OOM / timeout / restart.**
- **Day 8 (~5 h).** Inference of Magister-filter student on full test set. Compute accuracy.
- **Days 9–10 (~10 h).** Second full fine-tuning run (Set A / no filter). Inference. Accuracy.
- **Day 11 (~5 h).** Implement ReCEval scoring pipeline (NLI for correctness, log-likelihood LM for informativeness). Validate on 10 hand-checked outputs. Run on all three output sets.

### Week 3 — Audit, analysis, and write-up (~20 h)

- **Day 12 (~5 h).** Manual audit: 50 outputs, blind labeling.
- **Day 13 (~5 h).** Audit analysis: cross-tabulate, identify representative examples.
- **Days 14–15 (~10 h).** Write-up: results, discussion, limitations, future work. Generate plots and tables.

### Built-in slack

- **At least one full day** of slack is built in across the schedule for the inevitable failed run, free-tier session timeout, or library version mismatch.
- If Week 1 falls behind, drop Set A (the no-filter run) and report only baseline vs. Magister. The filter ablation is desirable but not essential — the dual-metric evaluation is the contribution.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Free-tier session times out mid-training | high | Save checkpoints every epoch to HuggingFace Hub; resume from latest. Use mixed precision. Consider Kaggle (longer sessions) over Colab. |
| OOM with FLAN-T5-base + batch size 8 | medium | Drop to batch size 4 with gradient accumulation. Use fp16. |
| Ho et al.'s released CoT JSON has a different schema than expected | medium | Day 3 inspection task; if schema is unworkable, fall back to GSM8K's own gold reasoning steps as a proxy "teacher" (note this clearly in limitations). |
| Answer-parsing logic is fragile across student output styles | high | Implement and unit-test on baseline outputs in Week 1; iterate before training. |
| ReCEval scoring is too slow on 1,319 examples × 3 conditions | medium | NLI batching; if needed, evaluate on a 500-example test subset, not the full test set. Document this clearly. |
| Manual audit reveals ReCEval is poorly correlated with human judgement on this data | low–medium | Reportable as a finding ("ReCEval has limited validity in this setting"). Does not break the project. |
| All distillation conditions produce broadly similar accuracy (small model, small dataset) | medium | The dual-metric story still holds — even if accuracy is flat, ReCEval may differentiate. |

---

## 9. Deliverables

1. **Code repository** (GitHub or GitLab): training scripts, ReCEval scoring scripts, evaluation notebooks, audit spreadsheet, README with reproduction instructions.
2. **Model checkpoints** for the two fine-tuned students (HuggingFace Hub, private).
3. **Output data:** generated CoTs from baseline + 2 students, with parsed answers and ReCEval scores, in JSON.
4. **Manual audit spreadsheet** with rubric, labels, and ReCEval scores per example.
5. **Final report** (~10–15 pages): introduction, related work, method, experiments, manual audit, discussion, limitations, future work.

---

## 10. Future Work (explicitly out of scope but worth noting in the write-up)

- ReCEval as a *filter* on teacher CoTs (rather than only as evaluation), comparing against the answer-correctness filter.
- Larger students (T5-large or Qwen-1.5B with LoRA) on the same dual-metric framework.
- Multiple datasets (StrategyQA, ASDiv) to test cross-domain robustness of the dual-metric finding.
- Multiple teacher models, to study whether teacher CoT quality (as measured by ReCEval) predicts student CoT quality.
- The pvi-based informativeness variant of ReCEval, replacing the log-likelihood approximation.

---

## References

- Magister, Mallinson, Adamek, Malmi, Severyn. *Teaching Small Language Models to Reason*. ACL 2023 (Short). https://aclanthology.org/2023.acl-short.151/
- Prasad, Saha, Zhou, Bansal. *ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness*. EMNLP 2023. https://arxiv.org/abs/2304.10703
- Ho, Schmid, Yun. *Large Language Models Are Reasoning Teachers*. ACL 2023. https://github.com/itsnamgyu/reasoning-teacher
- Wei et al. *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.
- Cobbe et al. *Training Verifiers to Solve Math Word Problems* (GSM8K). 2021.
- Laurer et al. DeBERTa-v3 NLI model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`.