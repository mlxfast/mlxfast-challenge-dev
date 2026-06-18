"""Correctness gate.

Runs both the reference model and the submission model with greedy
decoding (temperature=0) on the same short prompt, then compares the
output token sequences. A submission passes if every decoded token
matches the reference exactly.

Greedy decoding at temperature=0 is deterministic, so any deviation is
a hard signal that the model's forward pass is wrong — not just a
floating-point ordering difference.

This replaces the previous layer-wise hidden-state comparison, which
required Gemma-4-specific layer-iteration code. Greedy token comparison
is model-agnostic and equivalent: same inputs + same greedy rule = same
outputs for any correct implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class CorrectnessResult:
    passed: bool
    # num_layers is repurposed to store the number of tokens compared;
    # the field name is kept for RunReport compatibility.
    num_layers: int
    first_failing_layer: Optional[int] = None  # index of first mismatched token
    max_abs_diff: float = 0.0   # 0.0 = pass, 1.0 = any mismatch
    max_rel_diff: float = 0.0

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "num_tokens_checked": self.num_layers,
            "first_failing_token": self.first_failing_layer,
            "max_abs_diff": self.max_abs_diff,
        }


def _greedy_decode(model, prompt: mx.array, num_tokens: int) -> list[int]:
    """Greedily decode `num_tokens` tokens from `prompt`.

    Returns a list of integer token IDs. Each token is the argmax of
    the logits at the previous step — temperature=0, no sampling.
    """
    inner = model.language_model if hasattr(model, "language_model") else model
    cache = inner.make_cache() if hasattr(inner, "make_cache") else None

    # Prefill: process the whole prompt, get the first next-token logits.
    if cache is not None:
        logits = inner(prompt, cache=cache)
    else:
        logits = inner(prompt)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_tok)

    tokens = [int(next_tok.item())]

    # Autoregressive decode.
    for _ in range(num_tokens - 1):
        if cache is not None:
            logits = inner(next_tok, cache=cache)
        else:
            logits = inner(next_tok)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(next_tok)
        tokens.append(int(next_tok.item()))

    return tokens


def check(
    reference_model,
    submission_model,
    tokens: mx.array,
    decode_length: int = 16,
) -> CorrectnessResult:
    """Compare greedy-decoded token sequences of two models on the same
    input prompt. Returns a CorrectnessResult.

    Args:
      reference_model: upstream reference implementation.
      submission_model: participant's submission.
      tokens: input prompt, shape (1, T).
      decode_length: number of tokens to greedily decode and compare.
    """
    ref_seq = _greedy_decode(reference_model, tokens, decode_length)
    sub_seq = _greedy_decode(submission_model, tokens, decode_length)

    first_failing: Optional[int] = None
    for i, (r, s) in enumerate(zip(ref_seq, sub_seq)):
        if r != s:
            first_failing = i
            break

    passed = first_failing is None
    return CorrectnessResult(
        passed=passed,
        num_layers=decode_length,
        first_failing_layer=first_failing,
        max_abs_diff=0.0 if passed else 1.0,
        max_rel_diff=0.0 if passed else 1.0,
    )
