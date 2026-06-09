"""Linear layer — the primary compute target. MODIFIABLE.

This is the single biggest knob in the challenge. The Linear class
defined here is what every attention projection, every MLP gate/up/down,
and (if experts.py routes through it) every expert projection uses.

What you can change:
  - The __call__ method: the forward pass. The harness measures Metal
    bandwidth on whatever memory this path actually touches.
  - The __init__ method: how weights are stored on the module. If
    transform.py writes weights in a different layout, this is where
    you teach Linear how to find them.
  - The class hierarchy: subclass nn.Linear (default), or write
    something entirely new.

What you should NOT change:
  - The class name. mlx_lm.models.gemma4_text uses `nn.Linear` and
    `mlx.nn.Linear` to construct these layers. The frozen __init__.py
    patches the global `nn.Linear` symbol to this class at import time
    so the upstream model code picks it up automatically.

Baseline behavior: identical to mlx.nn.Linear. A participant who
changes nothing here will get exactly the upstream 4-bit performance.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Linear(nn.Linear):
    """Drop-in replacement for mlx.nn.Linear.

    Default: passes through to nn.Linear. Override __call__ to compute
    on a transformed weight representation, or override __init__ to
    consume a non-standard weight layout produced by transform.py.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x)


# The frozen __init__.py imports this module and assigns Linear to
# mlx.nn.Linear before any model is constructed. The process is
# dedicated to running this one model, so a global patch is safe.
_nn_Linear = Linear


def install() -> type[nn.Linear]:
    """Install this Linear as mlx.nn.Linear. Idempotent."""
    nn.Linear = _nn_Linear
    return _nn_Linear
