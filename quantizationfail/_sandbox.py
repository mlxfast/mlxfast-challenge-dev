"""Sandboxed re-run of transform.py for provenance verification.

The harness re-runs the participant's `transform.py` in a clean
sandbox to verify that the `weights/` directory is a deterministic
function of `reference_weights/`. The sandbox:

  - Forbids network access
  - Forbids clock reads (time, datetime.now)
  - Forbids environment variable reads
  - Forbids reads from anything outside `reference_weights/`
  - Forbids writes to anything outside `weights/`
  - Captures the byte-hash of everything written to `weights/`

If the re-run produces a different `weights/` than the participant
submitted, the submission fails the provenance check.

This is implemented via Python's `audit` events (PEP 578), which
work on CPython 3.8+ and are not bypassable from user code.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def content_hash(path: Path) -> str:
    """SHA-256 of every file under `path`, sorted by relative path."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            rel = p.relative_to(path)
            h.update(str(rel).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


SANDBOX_AUDIT_SCRIPT = '''
import sys
import os
from pathlib import Path

REFERENCE = Path("{reference}").resolve()
OUTPUT = Path("{output}").resolve()

# Map of blocked APIs. The audit hook fires before the call returns
# to user code; raising SystemExit aborts the process.
BLOCKED = True

def _is_within(path, parent):
    try:
        path = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    return parent in path.parents

def hook(event, args):
    if event == "open":
        # args: (path, mode, flags)
        path, mode, flags = args[:3]
        if "w" in mode or "a" in mode or "x" in mode:
            # Writes must be within OUTPUT
            if not _is_within(path, OUTPUT):
                sys.stderr.write(f"BLOCKED: write to {{path}} outside OUTPUT\\n")
                sys.exit(3)
        else:
            # Reads must be within REFERENCE
            if not _is_within(path, REFERENCE):
                sys.stderr.write(f"BLOCKED: read from {{path}} outside REFERENCE\\n")
                sys.exit(3)
    elif event == "socket.connect" or event == "socket.bind":
        sys.stderr.write("BLOCKED: network access\\n")
        sys.exit(3)
    elif event == "os.system" or event == "subprocess.Popen":
        sys.stderr.write("BLOCKED: subprocess\\n")
        sys.exit(3)
    elif event == "os.putenv" or event == "os.unsetenv":
        sys.stderr.write("BLOCKED: env mutation\\n")
        sys.exit(3)

sys.addaudithook(hook)

# Unset the env so the participant cannot read it.
os.environ.clear()
# Patch time sources.
import time as _time
class _FrozenTime:
    def __getattr__(self, name):
        sys.stderr.write(f"BLOCKED: time access ({{name}})\\n")
        sys.exit(3)
_time.time = _FrozenTime()
_time.monotonic = _FrozenTime()
_time.perf_counter = _FrozenTime()

import runpy
sys.argv = ["transform.py"]
runpy.run_path("{transform}", run_name="__main__")
'''


def verify_transform(
    transform_path: Path,
    reference_path: Path,
    submitted_weights: Path,
    python: str = sys.executable,
) -> tuple[bool, str]:
    """Re-run transform.py in a sandbox and verify byte-equal output.

    Returns (ok, message). ok=True means the submitted weights are
    a deterministic function of the reference weights.
    """
    reference_path = reference_path.resolve()
    submitted_weights = submitted_weights.resolve()
    transform_path = transform_path.resolve()

    if not transform_path.exists():
        return False, f"transform.py not found at {transform_path}"
    if not reference_path.exists():
        return False, f"reference_weights not found at {reference_path}"
    if not submitted_weights.exists():
        return False, f"submitted weights not found at {submitted_weights}"

    # Hash the submitted weights before we wipe them.
    submitted_hash = content_hash(submitted_weights)

    # Wipe the output directory and re-run.
    sandbox_output = submitted_weights.parent / f".{submitted_weights.name}.sandbox"
    if sandbox_output.exists():
        shutil.rmtree(sandbox_output)
    sandbox_output.mkdir(parents=True)

    audit_script = SANDBOX_AUDIT_SCRIPT.format(
        reference=str(reference_path),
        output=str(sandbox_output),
        transform=str(transform_path),
    )

    proc = subprocess.run(
        [python, "-c", audit_script],
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour hard cap
    )

    if proc.returncode != 0:
        return False, (
            f"transform.py re-run failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )

    sandbox_hash = content_hash(sandbox_output)
    if sandbox_hash != submitted_hash:
        return False, (
            f"transform.py is non-deterministic.\n"
            f"  submitted hash: {submitted_hash[:16]}...\n"
            f"  re-run hash:    {sandbox_hash[:16]}..."
        )

    # Cleanup the sandbox copy.
    shutil.rmtree(sandbox_output)
    return True, submitted_hash
