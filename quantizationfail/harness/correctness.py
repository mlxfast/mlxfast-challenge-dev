"""Correctness gate. FROZEN.

The challenge spec says: hidden states at every layer must match the
reference exactly up to floating point associativity. We implement
this as layer-wise comparison of the activations flowing out of
every DecoderLayer.

Implementation strategy:
  1. Load the reference model (mlx_lm's standard load, no modifiable
     surface involvement).
  2. Load the submission model (using the participant's modifiable
     surface).
  3. Run both on the same input tokens (seeded at runtime by the
     harness from a server-side secret + commit hash).
  4. At every DecoderLayer, capture the output hidden state.
  5. Compare with allclose using CORRECTNESS_EPSILON.

A pass means every layer is within tolerance. A fail reports the
first diverging layer and the magnitude of the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import mlx.core as mx

from .constants import CORRECTNESS_EPSILON


@dataclass
class CorrectnessResult:
    passed: bool
    num_layers: int
    first_failing_layer: Optional[int] = None
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    failing_layer_diffs: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "num_layers": self.num_layers,
            "first_failing_layer": self.first_failing_layer,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
        }


def _capture_intermediates(model, tokens: mx.array) -> List[mx.array]:
    """Run the model and capture the hidden state at the output of
    every DecoderLayer. We use forward hooks that read the input
    and output of each layer.

    Returns a list of length num_hidden_layers, where element i is
    the output of layer i (before the final norm/lm_head).
    """
    intermediates: List[mx.array] = []
    handles = []

    def make_hook(idx):
        def hook(module, inputs, output):
            # DecoderLayer returns (h, shared_kv, offset) or just h
            # depending on the model. The hidden state is the first
            # element.
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            intermediates.append((idx, h))

        return hook

    layers = model.layers
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        # Use a fresh cache to avoid KV state leaking between runs.
        cache = model.make_cache() if hasattr(model, "make_cache") else None
        # The model may be the top-level wrapper (gemma4.Model) or
        # the inner text model (gemma4_text.Model). Try both.
        if hasattr(model, "language_model"):
            inner = model.language_model
        else:
            inner = model
        if cache is not None:
            _ = inner(tokens, cache=cache)
        else:
            _ = inner(tokens)
        mx.eval([h for _, h in intermediates])
    finally:
        for h in handles:
            h.remove()

    intermediates.sort(key=lambda x: x[0])
    return [h for _, h in intermediates]


def check(
    reference_model,
    submission_model,
    tokens: mx.array,
    epsilon: float = CORRECTNESS_EPSILON,
) -> CorrectnessResult:
    """Compare layer-wise hidden states of two models on the same
    input tokens. Returns a CorrectnessResult.

    Both models must have the same num_hidden_layers and accept
    tokens of the same shape.
    """
    ref_intermediates = _capture_intermediates(reference_model, tokens)
    sub_intermediates = _capture_intermediates(submission_model, tokens)

    if len(ref_intermediates) != len(sub_intermediates):
        return CorrectnessResult(
            passed=False,
            num_layers=len(ref_intermediates),
            first_failing_layer=0,
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
        )

    num_layers = len(ref_intermediates)
    max_abs = 0.0
    max_rel = 0.0
    first_failing: Optional[int] = None
    failing_diffs: List[float] = []

    for i, (ref_h, sub_h) in enumerate(zip(ref_intermediates, sub_intermediates)):
        diff = mx.abs(ref_h - sub_h)
        abs_diff = float(mx.max(diff))
        # Relative diff: avoid div by zero on tiny reference values.
        ref_abs = mx.abs(ref_h)
        rel = mx.where(ref_abs > 1e-6, diff / ref_abs, mx.zeros_like(diff))
        rel_diff = float(mx.max(rel))

        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)

        if abs_diff > epsilon:
            if first_failing is None:
                first_failing = i
            failing_diffs.append(abs_diff)

    passed = first_failing is None
    return CorrectnessResult(
        passed=passed,
        num_layers=num_layers,
        first_failing_layer=first_failing,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        failing_layer_diffs=failing_diffs,
    )
