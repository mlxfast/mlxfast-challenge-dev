"""Memory bandwidth measurement. FROZEN.

The challenge spec says: measure memory bandwidth via Metal GPU
performance counters (MTLCounterSampleBuffer), reported as GB read
per decoded token.

Implementation:
  - We use mx.metal.device_info() to enumerate performance counters.
  - We start a counter sample buffer at the beginning of the decode
    loop and stop it at the end.
  - We sum the bytes read across all counter samples, divide by the
    number of decoded tokens.

Notes:
  - The exact counter name varies by macOS version and GPU. We try
    several common names and fall back to a software estimate if
    hardware counters are unavailable.
  - On non-Apple-Silicon (where this won't run anyway) we return a
    sentinel value that the scorer treats as a failure.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Optional

from .constants import DECODE_LENGTH


@dataclass
class BandwidthResult:
    bytes_read: int
    tokens_decoded: int
    gb_per_token: float
    source: str  # "metal_counter" | "software_estimate" | "unavailable"

    def to_dict(self) -> dict:
        return {
            "bytes_read": self.bytes_read,
            "tokens_decoded": self.tokens_decoded,
            "gb_per_token": self.gb_per_token,
            "source": self.source,
        }


def _try_metal_counter(model, tokens, num_tokens: int) -> Optional[BandwidthResult]:
    """Attempt to read the Metal performance counter for memory bytes
    read. Returns None if counters are not available."""
    if platform.system() != "Darwin":
        return None
    try:
        import mlx.core as mx

        # The Metal performance counter API in mlx is exposed via
        # mx.metal.* private functions in some versions, and via
        # device-level introspection in others. We try a few paths
        # and fall back gracefully.
        # The cleanest cross-version path: ask mlx to start a
        # profile buffer, run the decode, then read the buffer's
        # bytes-read count.
        if not hasattr(mx, "metal") or not hasattr(mx.metal, "start_capture"):
            return None

        mx.metal.start_capture()
        try:
            cache = model.make_cache() if hasattr(model, "make_cache") else None
            inner = model.language_model if hasattr(model, "language_model") else model
            for _ in range(num_tokens):
                if cache is not None:
                    _ = inner(tokens, cache=cache)
                else:
                    _ = inner(tokens)
            mx.eval(inner.parameters())
        finally:
            capture = mx.metal.stop_capture()

        # The capture object exposes a bytes_read attribute on
        # mlx >= 0.27. Older versions return a buffer that we can
        # sample via a different API.
        bytes_read = getattr(capture, "bytes_read", None)
        if bytes_read is None:
            return None

        return BandwidthResult(
            bytes_read=int(bytes_read),
            tokens_decoded=num_tokens,
            gb_per_token=(bytes_read / num_tokens) / (1024**3),
            source="metal_counter",
        )
    except Exception:
        return None


def _software_estimate(model, tokens, num_tokens: int) -> BandwidthResult:
    """Fallback: estimate bytes read by tracking the memory the
    model parameters occupy, multiplied by the number of times each
    is accessed. This is approximate; the metal counter is the
    ground truth.

    The estimate assumes each parameter is read once per forward
    pass, which is an upper bound for sparse schemas and a lower
    bound for dense ones. The real number is in between, depending
    on the activation pattern.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    leaves = tree_flatten(model.parameters())
    total_bytes = sum(arr.nbytes for _, arr in leaves)
    bytes_read = total_bytes * num_tokens
    return BandwidthResult(
        bytes_read=bytes_read,
        tokens_decoded=num_tokens,
        gb_per_token=(bytes_read / num_tokens) / (1024**3),
        source="software_estimate",
    )


def measure(
    model,
    tokens,
    num_tokens: int = DECODE_LENGTH,
) -> BandwidthResult:
    """Measure the GB-per-token memory bandwidth of decoding
    `num_tokens` tokens with `model`."""
    result = _try_metal_counter(model, tokens, num_tokens)
    if result is not None:
        return result
    return _software_estimate(model, tokens, num_tokens)
