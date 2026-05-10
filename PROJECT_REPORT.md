# Project Report — CoT Distillation + ReCEval at Small Scale

**Author:** Rihem Ben Abdallah

---

## 1. Introduction & Problem Statement

### Background

Chain-of-thought (CoT) prompting works well on large language models, but on small models (under ~10B parameters) it tends to produce fluent-looking but illogical reasoning, and can even hurt accuracy. **CoT knowledge distillation** is the standard workaround: a large teacher model generates step-by-step reasoning on a labelled dataset, the chains are filtered, and a small student is fine-tuned on the surviving `(question → CoT + answer)` pairs.

Two papers anchor my setup:

- **Ho et al. (ACL 2023)**, *Large Language Models Are Reasoning Teachers*. The paper I am reproducing. They report FLAN-T5-base (220M) on GSM8K at 2.50% zero-shot, 4.40% with CoT fine-tuning, and 5.08% with standard fine-tuning (Q → A only).
- **Magister et al. (2022)**, *Teaching Small Language Models to Reason*. The source of the **calculator rewrite-trick**: replace `A op B = C` substrings with the correct value, both at training-data construction time and during decoding. I use this to build Set C.

For a second evaluation axis I use **ReCEval** (Prasad et al., EMNLP 2023), a reference-free framework that scores reasoning chains on intra-step entailment, inter-step non-contradiction, and informativeness toward the gold answer.

### Problem Statement

The headline metric in CoT distillation work is **Exact Match accuracy** on the final answer. That metric is blind to *how* the answer was reached:

- A chain can land on the right answer through wrong arithmetic or missing premises.
- A student that emits a polished-looking chain may still be pattern-matching, not reasoning.
- A no-CoT model that just emits a number can score at parity with or above a model that produces a chain.

**Research question.** *“How well can small language models learn reasoning via CoT knowledge distillation from a larger teacher, under realistic resource constraints?”*

To make that question concrete, I score each student on both accuracy and ReCEval, and I run a **filter ablation** (no filter / answer-correctness / calculator-corrected) plus a **Direct FT** control that produces no reasoning at all.

---

## 2. Related Work

This project sits at the intersection of three threads in the small-LM reasoning literature.

### CoT prompting and its limits on small models

**Wei et al. (2022)** showed that prepending a few CoT exemplars unlocks step-by-step reasoning in large models, with the largest gains on harder problems (e.g. GSM8K). Two limitations from that paper directly motivate this project:

1. CoT prompting helps only at scale — small models produce fluent but illogical chains, and CoT prompting can actually *hurt* their accuracy.
2. There is no guarantee that the chain reflects correct reasoning, even when the final answer is right.


### Distilling reasoning into small models

**Magister et al. (2022)** propose the standard 2-step pipeline I follow: (1) prompt a large teacher (PaLM 540B / GPT-3 175B) to generate CoTs on an existing supervised dataset, keeping only chains whose final answer matches gold; (2) fine-tune a small student via teacher forcing with `(question → CoT + answer)`. They also introduce the **calculator rewrite-trick** for arithmetic tasks. Their headline number on GSM8K (T5-XXL improving from 8.11% to 21.99%) shows the method works at scale, but says little about a 220M-parameter student on free-tier hardware.

**Ho et al. (2023, ACL)** is the paper this project reproduces. Two extensions over Magister are relevant: (i) *zero-shot* prompting of the teacher (“Let’s think step by step”) instead of few-shot, and (ii) **diverse reasoning** — multiple distinct rationales per question to enrich the training set. They also report a finding I rely on: ~28% of teacher chains that pass the answer-correctness filter still contain incorrect intermediate reasoning. This is exactly the failure mode that motivated my Set C (calculator-corrected) ablation.

**ThinkSLM (2025)** is a more recent reference for what is achievable with modern small models (Qwen2.5-3B-Instruct hits ~84–85% on GSM8K). It is out of scope for the FLAN-T5-base setup I use here, but it argues usefully that reasoning ability in small LMs depends more on training recipe and data quality than on raw parameter count, which lines up with what my filter ablation measures.

### Evaluating reasoning, not just answers

**Prasad et al. (2023, ReCEval)** is the second axis of my evaluation. ReCEval is reference-free: instead of comparing a chain to a gold rationale (which GSM8K does not provide), it scores each step on intra-step entailment, inter-step non-contradiction, and informativeness toward the gold answer. I use this on the *student’s* generated chains, which is where it bites accuracy alone cannot tell me whether the student actually reasons or just guesses well.

### Where this project sits

I take Ho et al.’s setup (FLAN-T5-base, GSM8K, zero-shot teacher, pre-released CoT data from `itsnamgyu/reasoning-teacher`), keep Magister’s calculator for one of the filter ablations, and add ReCEval on top of accuracy. The novel piece is dual-metric comparison across a 5-condition matrix that includes a no-CoT control (Direct FT) designed to expose the gap between “gets the answer” and “reasons through the answer”.

---

## 3. Current Work and Progress

### Conditions matrix (all complete through Stage 4)

| # | Condition | Training data | Format | Reference |
|---|---|---|---|---|
| 1 | **Baseline** | none (zero-shot) | n/a | Ho et al. zero-shot 2.50% |
| 2 | **Direct FT** | 7,473 (Q, A) pairs, no CoT | `Q: {q}` → `#### {gold}` | Ho et al. standard FT 5.08% |
| 3 | **Set A** | 7,473 unfiltered teacher CoTs | `Q: {q}` → `{cot} #### {gold}` | Ho et al. CoT FT 4.40% |
| 4 | **Set B** | 3,389 chains, answer-correctness filter | same as A | Zelikman et al. (2022) |
| 5 | **Set C** | 2,635 chains, calculator-corrected filter | same as A | uses Magister 2022 calculator |

### Pipeline stages

| Stage | Script | Status | What was built |
|---|---|---|---|
| 1 | `01_download.sh` | done | GSM8K test set + Ho et al. teacher CoTs from `itsnamgyu/reasoning-teacher` |
| 2 | `02_filter.sh` | done | Set A / B / C / Direct FT JSONL files; B↔C contingency table written to run-card |
| 3 | `03_train_*.sh` | done (4 runs) | FLAN-T5-base fine-tuned with v2 recipe (lr=5e-5, wd=0.01, 8 epochs, max_len=512); early-stopped on val loss with patience 2 |
| 4 | `04_inference.sh` | done | Beam=4, `no_repeat_ngram_size=4`, `repetition_penalty=1.15`, `max_new_tokens=512` on full GSM8K test (1,319) for all 5 conditions |
| 5a | inline in notebook | done | Accuracy + accuracy-with-calculator computed from generation files |
| 5b | `05b_receval.sh` | **pending** | ReCEval scoring on all 5 conditions |
| 6 | `06_audit_prep.sh` | pending | 50-example blind audit spreadsheet |

### Stage 2 — filter statistics

```
GSM8K train rows ........... 7,473
Set A (no filter) .......... 7,473  (keep rate 100%)
Set B (answer-correct) ..... 3,389  (keep rate 45.4%)
Set C (calculator-correct) . 2,635  (keep rate 35.3%)
Direct FT (Q→A only) ....... 7,473  (keep rate 100%)
Calculator-edited chains ... 676
```

B ↔ C contingency:

|              | in B   | not in B |
|--------------|--------|----------|
| **in C**     | 2,585  | 50       |
| **not in C** | 50     | 4,034    |

Set C is *not* strictly broader than Set B. It rescues 50 chains that fail B but whose arithmetic-fix lands on gold, and evicts ~800 chains that pass B by coincidence (right answer through wrong arithmetic).

### Stage 3 — training health

| Condition | n_train | n_val | Best epoch | Best eval loss |
|---|---|---|---|---|
| `student_direct_ft` | 6,725 | 748 | 8.0 | 0.868 |
| `student_set_a` | 6,725 | 748 | 8.0 | 0.922 |
| `student_set_b` | 3,050 | 339 | 7.0 | 0.922 |
| `student_set_c` | 2,371 | 264 | 8.0 | 0.888 |

### Stage 4 / 5a — accuracy

GSM8K test set, n = 1,319. Same decoding flags across all conditions.

| Condition | Accuracy | Acc. + calculator | Format compliance | Median CoT chars |
|---|---|---|---|---|
| `baseline` | **4.32%** | 4.32% | 0% | 241 |
| `student_direct_ft` | **5.00%** | 5.00% | 100% | 7 |
| `student_set_a` | 2.50% | 2.50% | 100% | 308 |
| `student_set_b` | 3.03% | 3.03% | 99.85% | 263 |
| `student_set_c` | **3.18%** | 3.18% | 100% | 258 |

### Comparison vs Ho et al. references

| Setting | Ho et al. acc | Our condition | Our acc | Gap |
|---|---|---|---|---|
| Zero-shot (no FT) | 2.50% | `baseline` | 4.32% | **+1.82 pp** |
| Standard FT (Q → A only) | 5.08% | `student_direct_ft` | 5.00% | −0.08 pp |
| CoT FT | 4.40% | `student_set_a` | 2.50% | −1.90 pp |
| CoT FT | 4.40% | `student_set_b` | 3.03% | −1.37 pp |
| CoT FT | 4.40% | `student_set_c` | 3.18% | −1.22 pp |

**Observations.**

- Direct FT (5.00%) closely matches Ho et al.'s 5.08% standard-FT reference. The no-CoT recipe is correctly implemented.
- Baseline sits 1.82 pp above Ho et al.'s zero-shot reference. This is a parsing-leniency artifact, not real outperformance.
- All three CoT students sit 1.2–1.9 pp below the CoT-FT reference, and below the un-finetuned baseline. This is the central tension the project is designed to expose. ReCEval (Stage 5b) is what tells me whether reasoning quality nevertheless improved.
- Within the CoT students, the ranking **Set C > Set B > Set A** is exactly what the v2 plan predicted: the process-aware filter beats the answer-correctness filter beats no filter.

---

## 4. Limitations and Observations

### Parsing leniency

`parse_answer.py` falls back to "last number in the string" when `####` is absent. This inflates the un-finetuned baseline relative to Ho et al.'s likely stricter `#### N` parse. It does *not* affect the fine-tuned conditions, which all emit `####` at ≥ 99.85% rate.

### Calculator scope

Magister's calculator runs **at decode time**: the model emits `A op B =`, the calculator computes the correct result, and that result is forced into the decoded sequence before the next token is sampled. My calculator is **post-hoc only** — it rewrites finished chains for Set C training-data construction (Stage 2) and for the `accuracy_w_calc` metric (Stage 5a). I do not intervene during decoding.

The consequence shows up in the results: `accuracy_w_calc` is identical to `accuracy` for every condition. The post-hoc pass changes zero answers.

### Failure mode of CoT students

Looking at the qualitative outputs in [notebooks/results.ipynb](notebooks/results.ipynb), failures are dominated by **premise errors**, not arithmetic slips. The student picks the wrong inputs into the equation (e.g. "70 / 60 = 10" when the problem requires "0.7 × 110") and then computes correctly from there. A surface-level calculator pass cannot rescue this. This is precisely why Set C's training-data fix transfers only weakly to test-time behaviour.

### Direct FT artifact

Direct FT got the highest accuracy (5.00%) by emitting **no reasoning at all** — median output length is 7 characters, i.e. literally `#### N`. It pattern-matches questions to the model's pretrained answer prior. If accuracy were the only metric, the conclusion would be "skip CoT distillation entirely". That is the artefact the dual-metric framing is designed to expose.

---

## 5. Pending and Future Work

### Stage 5b — ReCEval scoring (next checkpoint)

For each of the 5 conditions × 1,319 generations:

1. Sentence-split the chain (spaCy, simplified-RCU: sentence = step).
2. **Intra-step** — entailment probability per step (NLI).
3. **Inter-step** — `1 − max P(contradiction)` between step *i* and prior context.
4. **Informativeness** — `log p(gold | Q, steps_1..i) − log p(gold | Q, steps_1..i-1)` under a frozen GPT-2.
5. Aggregate per-chain as **min over steps**.

**Open question this answers.** Does ReCEval rank the students differently than accuracy does? If Set C dominates on intra-step + informativeness despite trailing on accuracy, that is the cleanest version of the project's headline finding.

### Stage 6 — Manual audit

50 examples drawn from `student_set_b` outputs (the most representative system), stratified into 25 correct / 25 incorrect. Blind labels on a 1–5 reasoning-quality scale, then correlated with ReCEval scores to check whether the metric agrees with human judgement.

### Possible extensions

- **Online calculator decoding** (full Magister 2022 setup) — would close part of the gap to Ho et al.'s 4.40% CoT-FT reference.
- **FLAN-T5-small (60M) size ablation** — repeat conditions 1, 3, 4 to produce a FLOPs/accuracy curve.
- **StrategyQA** — second dataset to test whether the dual-metric story generalises beyond math.
- **ReCEval as a filter** (rather than only as evaluation) — compare against the answer-correctness filter.

---

## 6. References

1. **Ho, N., Schmid, L., & Yun, S.-Y.** (2023). *Large Language Models Are Reasoning Teachers*. ACL 2023. Code/data: [github.com/itsnamgyu/reasoning-teacher](https://github.com/itsnamgyu/reasoning-teacher)
2. **Magister, L. C., Mallinson, J., Adamek, J., Malmi, E., & Severyn, A.** (2022). *Teaching Small Language Models to Reason*. arXiv:2212.08410. (Source of the calculator rewrite-trick used in Set C.)
3. **Prasad, A., Saha, S., Zhou, X., & Bansal, M.** (2023). *ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness*. EMNLP 2023.
4. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D.** (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. NeurIPS 2022.
5. **Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D.** (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023.
6. **Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D.** (2022). *STaR: Bootstrapping Reasoning with Reasoning*. NeurIPS 2022. (Origin of answer-correctness filtering, used in Set B.)
7. **Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., et al.** (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168. (GSM8K dataset.)
8. **Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., et al.** (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416. (FLAN-T5 family.)
9. **ThinkSLM** (2025). *THINKSLM: Towards Reasoning in Small Language Models*. (Reference for what modern small instruction-tuned models achieve on the same benchmarks.)
