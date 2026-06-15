"""Memory bandwidth measurement. FROZEN.

Bandwidth is measured via mactop, which reads Apple's hardware IOReport
DRAM counters directly:

  mactop --headless --count N --interval 100 --format json

Each JSON line contains soc_metrics.dram_bw_combined_gbs — the
instantaneous DRAM read+write bandwidth at that sample in GB/s.

The harness starts mactop before the decode loop, collects samples
concurrently, and computes:

  gb_per_token = (mean_non_zero_gbps × decode_duration_s) / num_tokens

If mactop is not installed or fails to produce samples, bandwidth
measurement fails closed.
"""
from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

from .constants import DECODE_LENGTH

MACTOP_BINARY = os.environ.get("MACTOP_BINARY", "")
MACTOP_INTERVAL_MS = 100          # sample every 100 ms
MACTOP_MAX_SAMPLES = 600          # 60 s — enough to outlast any decode run


@dataclass
class BandwidthResult:
    bytes_read: int
    tokens_decoded: int
    gb_per_token: float
    source: str   # "mactop_hardware"

    def to_dict(self) -> dict:
        return {
            "bytes_read": self.bytes_read,
            "tokens_decoded": self.tokens_decoded,
            "gb_per_token": self.gb_per_token,
            "source": self.source,
        }


# ── mactop hardware measurement ───────────────────────────────────────────────

def _find_mactop_binary() -> Optional[str]:
    candidates = [
        MACTOP_BINARY,
        shutil.which("mactop"),
        "/opt/homebrew/bin/mactop",
        "/usr/local/bin/mactop",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


class MactopSession:
    """Context manager that runs mactop in the background and collects samples."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._samples: List[float] = []
        self._start: float = 0.0

    def start(self) -> bool:
        """Start mactop. Returns False if binary not found."""
        binary = _find_mactop_binary()
        if binary is None:
            return False
        try:
            self._proc = subprocess.Popen(
                [
                    binary,
                    "--headless",
                    "--count", str(MACTOP_MAX_SAMPLES),
                    "--interval", str(MACTOP_INTERVAL_MS),
                    "--format", "json",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._start = time.perf_counter()
            return True
        except OSError:
            return False

    def stop(self) -> List[float]:
        """Terminate mactop and return non-zero DRAM BW samples (GB/s)."""
        if self._proc is None:
            return []
        try:
            os.kill(self._proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, _ = self._proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            stdout, _ = self._proc.communicate()

        samples: List[float] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                bw = obj.get("soc_metrics", {}).get("dram_bw_combined_gbs", 0.0)
                if bw > 0.0:
                    samples.append(float(bw))
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
        return samples

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self._samples = self.stop()


def _mactop_result(session: MactopSession, num_tokens: int, decode_duration: float) -> Optional[BandwidthResult]:
    samples = session._samples
    if not samples:
        return None
    mean_gbps = sum(samples) / len(samples)
    total_gb = mean_gbps * decode_duration
    gb_per_token = total_gb / num_tokens if num_tokens > 0 else 0.0
    total_bytes = int(total_gb * (1024 ** 3))
    return BandwidthResult(
        bytes_read=total_bytes,
        tokens_decoded=num_tokens,
        gb_per_token=gb_per_token,
        source="mactop_hardware",
    )


# ── public API ────────────────────────────────────────────────────────────────

def measure(
    mactop_session: MactopSession,
    num_tokens: int = DECODE_LENGTH,
    decode_duration: float = 0.0,
) -> BandwidthResult:
    """Return mactop-measured bandwidth for `num_tokens` decode steps.

    The session must be started before the decode loop and stopped after.
    Missing or empty mactop samples are a hard failure.
    """
    if mactop_session is None:
        raise TypeError("mactop_session is required")

    result = _mactop_result(mactop_session, num_tokens, decode_duration)
    if result is None:
        raise RuntimeError(
            "mactop produced no non-zero DRAM bandwidth samples; "
            "install mactop and ensure hardware counters are available"
        )
    return result
