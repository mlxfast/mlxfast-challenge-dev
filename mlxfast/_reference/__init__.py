"""Frozen, trusted ground-truth reference runner.

Lives inside the (non-editable, hash-verified) ``mlxfast`` package — NOT in
the participant's editable ``mlx_models/`` surface — so a submission cannot
influence the ground truth it is scored against.

Pieces:
  - reference_bank.ReferenceSlotBank — streams experts straight from the
    original reference checkpoint's stacked safetensors (no offline transform).
  - reference_runner (added alongside) — loads a pinned-baseline model that
    streams via ReferenceSlotBank and produces ground-truth logits for the
    teacher-forced correctness comparison.
"""
from .reference_bank import ReferenceSlotBank

__all__ = ["ReferenceSlotBank"]
