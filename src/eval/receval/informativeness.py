"""Informativeness via causal LM conditional log-likelihood.

For each step i, compute:
    info_i = log p(gold | q, steps[:i+1]) − log p(gold | q, steps[:i])

where log p is the mean per-token log-probability of the gold answer tokens
under a frozen causal LM given the preceding context.

Default model: EleutherAI/pythia-410m (v3). Pass model_name="gpt2" to
reproduce v2 scores.

Chain score = min info_i over all steps. Negative values are expected when
adding a step makes the gold answer less predictable — this is the correct
behaviour, not a bug.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DEFAULT_MODEL = "distilgpt2"
_state: dict = {}


def init(model_name: str = _DEFAULT_MODEL) -> None:
    """Load (or reload) the scorer with a specific model."""
    if _state.get("model_name") == model_name:
        return
    _state.clear()
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(device).eval()
    max_len = (
        getattr(mdl.config, "n_positions", None)
        or getattr(mdl.config, "max_position_embeddings", 1024)
    )
    _state.update(model=mdl, tokenizer=tok, device=device,
                  max_len=max_len, model_name=model_name)


def _load() -> None:
    if "model" not in _state:
        init(_DEFAULT_MODEL)


@torch.no_grad()
def _log_p_suffix(context: str, suffix: str) -> float:
    """Mean log p(suffix tokens | context) under the loaded causal LM."""
    _load()
    model = _state["model"]
    tokenizer = _state["tokenizer"]
    device = _state["device"]
    max_len = _state["max_len"]

    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    sfx_ids = tokenizer.encode(" " + suffix.strip(), add_special_tokens=False)
    if not sfx_ids:
        return 0.0

    if len(ctx_ids) + len(sfx_ids) > max_len:
        keep = max(max_len - len(sfx_ids), 1)
        ctx_ids = ctx_ids[-keep:]

    full_ids = ctx_ids + sfx_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    logits = model(input_ids).logits[0]            # [seq, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)  # [seq, vocab]

    ctx_len = len(ctx_ids)
    sfx_tensor = torch.tensor(sfx_ids, dtype=torch.long, device=device)
    relevant = log_probs[ctx_len - 1 : ctx_len + len(sfx_ids) - 1]  # [sfx, vocab]
    token_lp = relevant.gather(1, sfx_tensor.unsqueeze(1)).squeeze(1)
    return token_lp.mean().item()


def score_chain(question: str, steps: list[str], gold_answer) -> float:
    """Return min incremental informativeness across steps."""
    if not steps:
        return float("nan")
    gold_str = f"{gold_answer:g}" if isinstance(gold_answer, float) else str(gold_answer)

    contexts = [question] + [
        question + " " + " ".join(steps[: i + 1]) for i in range(len(steps))
    ]
    log_ps = [_log_p_suffix(ctx, gold_str) for ctx in contexts]
    info_gains = [log_ps[i + 1] - log_ps[i] for i in range(len(steps))]
    return min(info_gains)


def score_steps(question: str, steps: list[str], gold_answer) -> list[float]:
    """Return per-step info gains for detailed inspection."""
    if not steps:
        return []
    gold_str = f"{gold_answer:g}" if isinstance(gold_answer, float) else str(gold_answer)
    contexts = [question] + [
        question + " " + " ".join(steps[: i + 1]) for i in range(len(steps))
    ]
    log_ps = [_log_p_suffix(ctx, gold_str) for ctx in contexts]
    return [log_ps[i + 1] - log_ps[i] for i in range(len(steps))]
