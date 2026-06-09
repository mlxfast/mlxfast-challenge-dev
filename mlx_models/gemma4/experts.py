"""Expert MLP implementation. MODIFIABLE.

The MoE block in Gemma 4 26B-A4B is a SwitchGLU: a stack of expert
matrices (num_experts, output_dims, input_dims) where each token's
routing indices select which expert(s) it reads. The compute pattern
is a sparse gather-matmul:

    y[token, i] = sum_k weight[top_k_indices[token, k], i, :] * x[token, :]

This is the dominant bandwidth target in the model: 3.8B activated
parameters per token, streamed through memory on every forward pass.

What you can change:
  - SwitchGLU: how the gate/up/down projections compose. Default is
    pass-through to upstream.
  - QuantizedSwitchLinear.__call__: the actual gather-matmul. The
    default calls mx.gather_qmm. Override to read a different layout
    from self.weight (which is what transform.py will produce).
  - The Router: subclass the upstream Router to change routing, but
    note that routing changes that alter which experts are activated
    will fail the layer-wise correctness gate.

What you should NOT change:
  - The class names. The frozen __init__.py patches
    mlx_lm.models.switch_layers.SwitchGLU, SwitchLinear, and
    QuantizedSwitchLinear with these classes at import time.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear as _UpstreamQuantizedSwitchLinear,
    SwitchGLU as _UpstreamSwitchGLU,
    SwitchLinear as _UpstreamSwitchLinear,
)


class SwitchLinear(_UpstreamSwitchLinear):
    """fp16/fp32 expert linear. Default: pass-through to upstream.

    Override __call__ to read from a different weight layout.
    The upstream version calls mx.gather_mm, which gathers expert
    rows and does a batched matmul.
    """

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool = False,
    ) -> mx.array:
        return super().__call__(x, indices, sorted_indices=sorted_indices)


class QuantizedSwitchLinear(_UpstreamQuantizedSwitchLinear):
    """4-bit quantized expert linear — the primary bandwidth target.

    Default: pass-through to upstream. Upstream calls mx.gather_qmm
    with self.weight (packed), self.scales, self.biases. The packed
    weight shape is (num_experts, output_dims_packed, input_dims,
    group_size_packed).

    To implement a suffix-sum or permutation-based schema:
      1. Have transform.py write a different weight layout (e.g.,
         suffix-summable, block-sparse, etc.) and store it in
         self.weight (or in additional buffers).
      2. Override __call__ to compute against that layout.
      3. The Metal counter on the harness will measure whatever
         memory this __call__ actually reads.
    """

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        sorted_indices: bool = False,
    ) -> mx.array:
        return super().__call__(x, indices, sorted_indices=sorted_indices)


class SwitchGLU(_UpstreamSwitchGLU):
    """Expert MLP block: gate_proj, up_proj, down_proj with an activation.

    Default: pass-through. Override to use a different activation
    (e.g., GeGLU vs SwiGLU) or to compose the projections differently.
    """

    pass


def install() -> None:
    """Install our SwitchGLU/SwitchLinear/QuantizedSwitchLinear on
    mlx_lm.models.switch_layers. Idempotent. The frozen __init__.py
    calls this at import time before any model is constructed."""
    import mlx_lm.models.switch_layers as _switch_layers

    _switch_layers.SwitchGLU = SwitchGLU
    _switch_layers.SwitchLinear = SwitchLinear
    _switch_layers.QuantizedSwitchLinear = QuantizedSwitchLinear
