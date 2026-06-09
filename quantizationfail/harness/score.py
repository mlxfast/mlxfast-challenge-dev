"""Score computation. FROZEN.

score = peak_ram_GB * bandwidth_GB_per_token * seconds_per_token

All three axes are measured independently by the harness. The score
is a derived quantity. Correctness is a hard gate — failing
submissions are not scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreResult:
    peak_ram_gb: float
    bandwidth_gb_per_token: float
    seconds_per_token: float
    score: float
    passed_correctness: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "peak_ram_gb": self.peak_ram_gb,
            "bandwidth_gb_per_token": self.bandwidth_gb_per_token,
            "seconds_per_token": self.seconds_per_token,
            "score": self.score,
            "passed_correctness": self.passed_correctness,
            "note": self.note,
        }


def compute(
    peak_ram_bytes: int,
    bandwidth_gb_per_token: float,
    seconds_per_token: float,
    passed_correctness: bool,
    note: str = "",
) -> ScoreResult:
    """Compute the score from measured quantities.

    A submission that fails correctness gets score=inf so the
    leaderboard sorts it to the bottom and the CLI can show a
    clear failure.
    """
    peak_ram_gb = peak_ram_bytes / (1024**3)
    if not passed_correctness:
        score = float("inf")
    else:
        score = peak_ram_gb * bandwidth_gb_per_token * seconds_per_token
    return ScoreResult(
        peak_ram_gb=peak_ram_gb,
        bandwidth_gb_per_token=bandwidth_gb_per_token,
        seconds_per_token=seconds_per_token,
        score=score,
        passed_correctness=passed_correctness,
        note=note,
    )
