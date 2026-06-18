"""Main harness entry point. FROZEN.

This is the function the CLI's `mlxfast run` calls. It:

  1. Verifies the modifiable surface exists and is loadable.
  2. Loads the reference model (no modifiable surface involvement).
  3. Loads the submission model (using the modifiable surface).
  4. Loads two real-text prompts from files or CI env vars.
  5. Runs the correctness gate (greedy token comparison, 64-token prompt).
  6. Measures peak RAM, bandwidth, decode latency (64-token context).
  7. Measures prefill latency as the cost of a single 32k-token context fill.
  8. Computes the score.
  9. Returns a RunReport that the CLI appends to results.tsv.

The harness runs in a subprocess (started by _harness_runner.py)
so that peak RAM is measured via the subprocess's resident set size
rather than the parent CLI's.

Prompts
-------
Two text files are required:
  prompts/correctness_local.txt  — short (CORRECTNESS_PROMPT_TOKENS tokens)
  prompts/benchmark_local.txt    — long  (BENCHMARK_PROMPT_TOKENS tokens)

The server overrides these via env vars:
  MLXFAST_CORRECTNESS_PROMPT  — UTF-8 text, same token length
  MLXFAST_BENCHMARK_PROMPT    — UTF-8 text, same token length

Both local and server versions tokenise to the same length but have
different content, so hardcoded outputs fail server-side validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

from . import bandwidth, constants, correctness, score


@dataclass
class RunReport:
    timestamp: str
    git_commit: str
    note: str
    peak_ram_gb: float
    bandwidth_gb_per_token: float
    decode_seconds_per_token: float
    prefill_seconds_per_token: float
    score: float
    passed_correctness: bool
    num_layers: int
    first_failing_layer: Optional[int]
    max_abs_diff: float
    harness_hash: str
    mlx_version: str
    mlx_lm_version: str
    error: Optional[str] = None
    bandwidth_source: str = ""

    def to_tsv_row(self) -> str:
        return "\t".join(
            [
                self.timestamp,
                self.git_commit,
                self.note,
                f"{self.peak_ram_gb:.4f}",
                f"{self.bandwidth_gb_per_token:.6f}",
                f"{self.decode_seconds_per_token:.6f}",
                f"{self.prefill_seconds_per_token:.6f}",
                f"{self.score:.6f}" if self.score != float("inf") else "inf",
                "1" if self.passed_correctness else "0",
                str(self.num_layers),
                str(self.first_failing_layer) if self.first_failing_layer is not None else "-",
                f"{self.max_abs_diff:.6e}",
                self.bandwidth_source,
                self.harness_hash[:12],
            ]
        )

    @staticmethod
    def tsv_header() -> str:
        return "\t".join(
            [
                "timestamp",
                "commit",
                "note",
                "peak_ram_gb",
                "bandwidth_gb_per_tok",
                "decode_sec_per_tok",
                "prefill_sec_per_tok",
                "score",
                "passed",
                "num_layers",
                "first_failing_layer",
                "max_abs_diff",
                "bandwidth_source",
                "harness_hash",
            ]
        )


def _get_git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def _versions() -> tuple[str, str]:
    try:
        import mlx
        mlx_v = mlx.__version__
    except Exception:
        mlx_v = "unknown"
    try:
        import mlx_lm
        mlx_lm_v = mlx_lm.__version__
    except Exception:
        mlx_lm_v = "unknown"
    return mlx_v, mlx_lm_v


def _load_prompt_text(env_var: str, local_path: Path) -> str:
    """Return prompt text from a CI env var (server) or local file (dev).

    The server sets the env var to its private prompt text so participants
    cannot predict the input. Locally, the committed file is used instead.
    """
    text = os.environ.get(env_var, "").strip()
    if text:
        return text
    if local_path.exists():
        return local_path.read_text().strip()
    raise FileNotFoundError(
        f"No prompt found: set {env_var} env var or provide {local_path}. "
        f"Run scripts/make_prompts.py to generate local prompt files."
    )


def _tokenize(tokenizer, text: str, max_len: int) -> mx.array:
    """Tokenize `text`, truncate to `max_len` tokens, return shape (1, max_len).

    Raises if the text tokenises to fewer than max_len tokens — the prompt
    files must be long enough. Run scripts/make_prompts.py to regenerate.
    """
    ids = tokenizer.encode(text)
    if len(ids) < max_len:
        raise ValueError(
            f"Prompt too short after tokenisation: {len(ids)} tokens, "
            f"need at least {max_len}. Run scripts/make_prompts.py."
        )
    return mx.array(ids[:max_len], dtype=mx.int32)[None]


def _peak_gpu_memory_gb() -> float:
    """Return peak GPU memory usage in GB.

    Uses mx.get_peak_memory() which tracks GPU memory allocations
    through MLX's caching allocator. This is the standard approach
    used by mlx_lm.stream_generate() and mlx_lm.benchmark.py.

    The caller must call mx.reset_peak_memory() before the measured
    section and mx.get_peak_memory() after eval completes.
    """
    return mx.get_peak_memory() / (1024**3)


def _install_participant_layer_patches() -> None:
    """Install the participant's nn.Linear and expert layer patches.

    Loads linear.py and experts.py directly (bypassing
    mlx_models/gemma4/__init__.py) so that only the layer-level
    primitives are patched — nn.Linear and switch_layers classes —
    without replacing mlx_lm.models.gemma4_text.Model.

    Why bypass __init__.py:
      __init__.py also does `gemma4_text.Model = participant_Model`,
      where participant_Model inherits from the OUTER gemma4.Model.
      When the outer model's __init__ later calls
      `gemma4_text.Model(inner_args)`, it would call the participant's
      outer model with inner ModelArgs (no text_config field) and
      raise AttributeError. Loading linear.py and experts.py directly
      avoids this, while still ensuring every nn.Linear and expert
      layer constructed during model building uses the participant's
      classes.
    """
    import importlib.util as _ilu
    from pathlib import Path as _Path

    _moddir = _Path("mlx_models/gemma4")
    for _name, _fname in [
        ("_qf_linear", "linear.py"),
        ("_qf_experts", "experts.py"),
    ]:
        _spec = _ilu.spec_from_file_location(_name, _moddir / _fname)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.install()


def _load_models(weights_path: Path):
    """Load the reference and submission models.

    Load order matters for the global nn.Linear / switch_layers patches:

      1. Load the reference model FIRST, before any patches are applied.
         The reference model's layer instances capture the upstream
         nn.Linear class at construction time; patching nn.Linear
         afterwards does not affect already-constructed instances.

      2. Install the participant's layer-level patches (nn.Linear,
         QuantizedSwitchLinear, SwitchGLU). These are process-global
         patches that affect all subsequently constructed layers.

      3. Load the submission model. All newly constructed nn.Linear
         and expert instances will use the participant's classes.
         The participant's Model class is loaded via the weights/
         config.json `model_file` pointer by mlx_lm.load — no separate
         Model class patching is needed.

    This ensures the reference model always uses the upstream
    implementation and the submission model always uses the
    participant's modifiable surface, even when both are loaded in
    the same process.
    """
    import mlx_lm

    # Step 1 — reference model (upstream, no patches).
    ref_model, ref_tokenizer = mlx_lm.load(
        str(constants.REFERENCE_WEIGHTS_DIR / constants.REFERENCE_MODEL_DIRNAME)
    )
    ref_model.eval()
    mx.eval(ref_model.parameters())

    # Step 2 — install participant's layer-level patches.
    # Patching is done before the submission model is constructed so
    # every new nn.Linear / expert layer uses the participant's class.
    _install_participant_layer_patches()

    # Step 3 — submission model.
    # mlx_lm.load reads weights/config.json; the `model_file` key
    # there points mlx_lm at the participant's model.py for the Model
    # and ModelArgs classes.
    sub_model, sub_tokenizer = mlx_lm.load(str(weights_path))
    sub_model.eval()
    mx.eval(sub_model.parameters())

    return ref_model, ref_tokenizer, sub_model


def _load_config_dict(weights_path: Path) -> dict:
    config_path = weights_path / "config.json"
    with open(config_path) as f:
        return json.load(f)


def _measure_latency_and_memory(model, prompt: mx.array, num_tokens: int) -> tuple[float, float]:
    """Decode `num_tokens` tokens, return (seconds_per_token, peak_ram_gb).

    Resets the MLX peak memory counter before the decode loop, then
    reads peak memory after eval. This ensures peak RAM captures only
    the decode phase, not prefill or warmup.
    """
    cache = model.make_cache() if hasattr(model, "make_cache") else None
    inner = model.language_model if hasattr(model, "language_model") else model

    # Warmup
    if cache is not None:
        _ = inner(prompt, cache=cache)
    else:
        _ = inner(prompt)
    mx.eval(inner.parameters())

    # Reset peak memory counter before measuring decode.
    mx.reset_peak_memory()

    # Decode loop
    t0 = time.perf_counter()
    next_tok = prompt[-1:]  # seed
    for _ in range(num_tokens):
        if cache is not None:
            logits = inner(next_tok, cache=cache)
        else:
            logits = inner(next_tok)
        # Greedy argmax
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    mx.eval(next_tok)
    elapsed = time.perf_counter() - t0

    seconds_per_token = elapsed / num_tokens
    peak_ram_gb = _peak_gpu_memory_gb()

    return seconds_per_token, peak_ram_gb


def _measure_prefill_latency(model, prompt: mx.array) -> float:
    """Measure prefill seconds-per-token for a full 32k-token context fill.

    Prefill processes the entire prompt in one batched forward call.
    The cost is reported as wall-clock seconds divided by the prompt
    length, so it is directly comparable across submissions regardless
    of prompt length changes.

    A single warmup pass runs first (JIT compile + weight cache warm-up),
    then one timed pass. A second timed pass is omitted because at 32k
    tokens the run time is substantial and variance is low.

    Returns:
      Wall-clock seconds per prompt token for the timed pass.
    """
    inner = model.language_model if hasattr(model, "language_model") else model
    prompt_len = prompt.shape[-1]

    def _prefill_once():
        cache = inner.make_cache() if hasattr(inner, "make_cache") else None
        if cache is not None:
            out = inner(prompt, cache=cache)
        else:
            out = inner(prompt)
        mx.eval(out)

    # Warmup — JIT compile, fill weight cache.
    _prefill_once()

    # Single timed pass — cost of filling the full 32k context.
    t0 = time.perf_counter()
    _prefill_once()
    elapsed = time.perf_counter() - t0

    return elapsed / prompt_len


def run(weights_path: Path, note: str) -> RunReport:
    """The main harness entry point."""
    import datetime

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    commit = _get_git_commit()
    mlx_v, mlx_lm_v = _versions()
    harness_hash = constants.compute_harness_hash()

    try:
        ref_model, ref_tokenizer, sub_model = _load_models(weights_path)

        # Load prompt text from CI env vars (server) or local files (dev).
        correctness_text = _load_prompt_text(
            constants.ENV_CORRECTNESS_PROMPT,
            constants.CORRECTNESS_PROMPT_FILE,
        )
        benchmark_text = _load_prompt_text(
            constants.ENV_BENCHMARK_PROMPT,
            constants.BENCHMARK_PROMPT_FILE,
        )

        # Tokenize. Each call truncates to the declared token count and
        # raises if the source text is too short.
        correctness_tokens = _tokenize(
            ref_tokenizer, correctness_text, constants.CORRECTNESS_PROMPT_TOKENS
        )
        benchmark_tokens = _tokenize(
            ref_tokenizer, benchmark_text, constants.BENCHMARK_PROMPT_TOKENS
        )

        # Correctness gate: greedy token comparison on the short prompt.
        # Fails fast — a model that doesn't pass gets scored as inf.
        correctness_result = correctness.check(
            ref_model, sub_model, correctness_tokens, decode_length=16
        )

        if not correctness_result.passed:
            return RunReport(
                timestamp=timestamp,
                git_commit=commit,
                note=note,
                peak_ram_gb=0.0,
                bandwidth_gb_per_token=0.0,
                decode_seconds_per_token=0.0,
                prefill_seconds_per_token=0.0,
                score=float("inf"),
                passed_correctness=False,
                num_layers=correctness_result.num_layers,
                first_failing_layer=correctness_result.first_failing_layer,
                max_abs_diff=correctness_result.max_abs_diff,
                harness_hash=harness_hash,
                mlx_version=mlx_v,
                mlx_lm_version=mlx_lm_v,
            )

        # Decode measurement uses the short correctness prompt as context.
        bw = bandwidth.measure(sub_model, correctness_tokens, constants.DECODE_LENGTH)
        decode_spt, peak_gb = _measure_latency_and_memory(
            sub_model, correctness_tokens, constants.DECODE_LENGTH
        )
        peak_bytes = int(peak_gb * (1024**3))

        # Prefill measurement: cost of filling the full 32k benchmark context.
        prefill_spt = _measure_prefill_latency(sub_model, benchmark_tokens)

        sr = score.compute(
            peak_ram_bytes=peak_bytes,
            bandwidth_gb_per_token=bw.gb_per_token,
            decode_seconds_per_token=decode_spt,
            prefill_seconds_per_token=prefill_spt,
            passed_correctness=True,
            note=note,
        )

        return RunReport(
            timestamp=timestamp,
            git_commit=commit,
            note=note,
            peak_ram_gb=sr.peak_ram_gb,
            bandwidth_gb_per_token=sr.bandwidth_gb_per_token,
            decode_seconds_per_token=sr.decode_seconds_per_token,
            prefill_seconds_per_token=sr.prefill_seconds_per_token,
            score=sr.score,
            passed_correctness=True,
            num_layers=correctness_result.num_layers,
            first_failing_layer=None,
            max_abs_diff=correctness_result.max_abs_diff,
            harness_hash=harness_hash,
            mlx_version=mlx_v,
            mlx_lm_version=mlx_lm_v,
            bandwidth_source=bw.source,
        )
    except Exception as e:
        return RunReport(
            timestamp=timestamp,
            git_commit=commit,
            note=note,
            peak_ram_gb=0.0,
            bandwidth_gb_per_token=0.0,
            decode_seconds_per_token=0.0,
            prefill_seconds_per_token=0.0,
            score=float("inf"),
            passed_correctness=False,
            num_layers=0,
            first_failing_layer=None,
            max_abs_diff=0.0,
            harness_hash=harness_hash,
            mlx_version=mlx_v,
            mlx_lm_version=mlx_lm_v,
            error=str(e),
        )


def main():
    parser = argparse.ArgumentParser(description="mlxfast harness")
    parser.add_argument("--weights", type=Path, default=constants.PARTICIPANT_WEIGHTS_DIR)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument(
        "--output-tsv", type=Path, default=constants.RESULTS_FILE
    )
    args = parser.parse_args()

    report = run(args.weights, args.note)

    # Append to results.tsv.
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if not args.output_tsv.exists():
        args.output_tsv.write_text(RunReport.tsv_header() + "\n")
    with open(args.output_tsv, "a") as f:
        f.write(report.to_tsv_row() + "\n")

    # Print a one-line summary.
    status = "PASS" if report.passed_correctness else "FAIL"
    print(
        f"[{status}] score={report.score:.4f} "
        f"ram={report.peak_ram_gb:.2f}GB "
        f"bw={report.bandwidth_gb_per_token:.4f}GB/tok "
        f"decode={report.decode_seconds_per_token:.4f}s/tok "
        f"prefill={report.prefill_seconds_per_token:.4f}s/tok"
    )
    if not report.passed_correctness:
        if report.first_failing_layer is not None:
            print(
                f"  first failing layer: {report.first_failing_layer} "
                f"(max_abs_diff={report.max_abs_diff:.3e})"
            )
        if report.error:
            print(f"  error: {report.error}")
    sys.exit(0 if report.passed_correctness else 1)


if __name__ == "__main__":
    main()
