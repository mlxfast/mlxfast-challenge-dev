# Prompt Design Spec

**Status:** Proposed
**Replaces:** `_seed_prompt()` random-token generation in `harness/run.py`

---

## Motivation

The current harness generates prompts by sampling random token IDs from a seeded RNG. This has two problems:

1. **Random tokens are out-of-distribution.** Real inference serves natural language. A model whose expert routing collapses on garbage input may pass correctness while failing silently on real text.
2. **No anti-cheat anchor.** Because the local and server prompts are derived from the same algorithm, a participant who reverse-engineers the seed could hardcode expected outputs.

This spec replaces random-token prompts with real text prompts that have a **local copy** (committed to the repo, visible to participants) and a **server copy** (never revealed, injected by CI). Both copies share the same token length so benchmark measurements are comparable, but their content differs so hardcoded outputs fail server-side validation.

---

## Two-prompt system

### Correctness prompt (short)

Used for the correctness gate. Short enough to run quickly but long enough to exercise routing across all 43 MoE layers.

- **Purpose:** Prove the submission model produces the same greedy-decoded token sequence as the reference model.
- **Decoding:** Temperature = 0 (greedy argmax). Deterministic — any deviation from the reference is a hard failure.
- **Length:** `CORRECTNESS_PROMPT_TOKENS = 64` tokens (after tokenization). Chosen to hit all layers without dominating wall time.
- **Check:** Compare the full output token sequence for `DECODE_LENGTH` steps against the reference model's output. All tokens must match exactly (greedy + deterministic = exact reproducibility).
- **Local file:** `prompts/correctness_local.txt` — committed to repo, participants can read and run against it.
- **Server file:** Injected by CI as `MLXFAST_CORRECTNESS_PROMPT` env var (UTF-8 text). Never printed, never logged. Only the pass/fail result is reported.

### Benchmark prompt (long)

Used for the bandwidth, decode latency, and prefill latency measurements.

- **Purpose:** Drive realistic inference load. Content exercises diverse expert routing paths.
- **Length:** `BENCHMARK_PROMPT_TOKENS = 32768` tokens (32k context window).
- **Local file:** `prompts/benchmark_local.txt` — committed to repo.
- **Server file:** Injected by CI as `MLXFAST_BENCHMARK_PROMPT` env var.

---

## File layout

```
prompts/
  correctness_local.txt   # Short prompt, local copy. Committed to repo.
  benchmark_local.txt     # Long prompt, local copy. Committed to repo.
```

Server copies are never stored in the repo. They are injected at CI time as environment variables.

The local and server files must satisfy:
- **Same token count** after tokenization with the model's tokenizer.
- **Different content** — no shared n-gram sequences longer than ~8 tokens.
- **Natural language** — prose, code, or structured data that a real user might submit.

---

## Anti-cheat guarantee

A submission that hardcodes outputs for the local prompts will fail the server-side correctness gate because:

1. The server prompt has different content → different hidden states at every layer → different logits → different greedy output tokens.
2. The correctness gate compares output tokens exactly (temp=0 is deterministic), so any hardcoded token sequence will diverge at the first differing position.
3. The server prompt is never revealed, logged, or included in the RunReport, so a participant cannot recover it from CI output.

This guarantee holds as long as the server prompts are rotated before any participant can brute-force them (they are arbitrary UTF-8 text, not algorithmic).

---

## Harness integration

### Loading prompts

```python
def _load_prompt_text(env_var: str, local_path: Path) -> str:
    """Return prompt text from env var (server/CI) or local file (dev)."""
    text = os.environ.get(env_var, "").strip()
    if text:
        return text
    if local_path.exists():
        return local_path.read_text().strip()
    raise FileNotFoundError(
        f"No prompt found: set {env_var} or provide {local_path}"
    )
```

### Tokenizing prompts

Prompts are tokenized with the model's tokenizer at harness startup. The resulting token IDs replace the current `_seed_prompt()` call.

```python
correctness_text = _load_prompt_text(
    "MLXFAST_CORRECTNESS_PROMPT",
    Path("prompts/correctness_local.txt"),
)
benchmark_text = _load_prompt_text(
    "MLXFAST_BENCHMARK_PROMPT",
    Path("prompts/benchmark_local.txt"),
)

# Tokenize — shape (1, T)
correctness_tokens = _tokenize(ref_tokenizer, correctness_text, CORRECTNESS_PROMPT_TOKENS)
benchmark_tokens   = _tokenize(ref_tokenizer, benchmark_text,   BENCHMARK_PROMPT_TOKENS)  # 32k tokens
```

```python
def _tokenize(tokenizer, text: str, expected_len: int) -> mx.array:
    """Tokenize text, assert length matches spec, return (1, T) array."""
    ids = tokenizer.encode(text)
    if len(ids) != expected_len:
        raise ValueError(
            f"Prompt tokenizes to {len(ids)} tokens; expected {expected_len}. "
            "Local and server prompts must be pre-validated to the same length."
        )
    return mx.array(ids, dtype=mx.int32)[None]
```

### Correctness check

The correctness gate shifts from layer-wise hidden-state comparison (current) to **greedy output token comparison**:

1. Run reference model on `correctness_tokens`, greedy-decode `DECODE_LENGTH` tokens, collect token IDs.
2. Run submission model on `correctness_tokens`, greedy-decode `DECODE_LENGTH` tokens, collect token IDs.
3. Compare sequences. All tokens must match.

This is strictly equivalent for a correctly implemented model (same inputs + temp=0 = same outputs), but:
- Faster than capturing all layer intermediates.
- Easier to reason about: "same tokens" is the observable contract.
- Avoids Gemma-4-specific layer-iteration code in `correctness.py`.

**Coherence sanity check (optional, non-blocking):** After the greedy run, verify that the output does not consist entirely of repeated tokens or padding IDs. This catches degenerate submissions that produce syntactically valid but semantically broken output. Logged as a warning, not a hard failure.

### Benchmark measurement

Replace the current `prefill_prompt` (which uses `seed ^ 0xDEADBEEF`) with `benchmark_tokens`. The decode seed prompt (`PROMPT_SEED_PREFIX_LENGTH = 32`) is also replaced by the first 32 tokens of `benchmark_tokens` to eliminate the remaining random-token surface.

---

## Constant changes

In `constants.py`:

| Old | New | Value |
|-----|-----|-------|
| `PROMPT_SEED_PREFIX_LENGTH = 32` | `CORRECTNESS_PROMPT_TOKENS = 64` | 64 |
| `PREFILL_PROMPT_LENGTH = 512` | `BENCHMARK_PROMPT_TOKENS = 32768` | 32768 |

`DECODE_LENGTH = 512` is unchanged.

---

## Prompt authoring guidelines

When creating the local and server prompt pairs:

1. **Pick two texts from the same domain** (e.g., both are long-form prose, code files, or concatenated Wikipedia articles). Similar domain → similar routing distribution → comparable benchmark numbers across local and server runs.
2. **Tokenize both with the model tokenizer** and pad/trim to exactly `CORRECTNESS_PROMPT_TOKENS` (64) and `BENCHMARK_PROMPT_TOKENS` (32768) before committing. At ~3.5 chars/token, the benchmark text source is roughly 115k characters (~20 pages of dense prose or a medium-sized source file).
3. **Concatenation is fine for the 32k prompt.** Stitch together multiple documents with a separator token (`\n\n---\n\n`) to reach the target length. Both local and server versions should use the same number and rough length of constituent documents so routing statistics are similar.
4. **Verify no shared long n-grams.** The longest common token subsequence between local and server benchmark prompts should be under 16 tokens (excluding separator tokens).
5. **Do not include personally identifiable information** in either file.
6. **Server prompts must be stored outside the repo** (e.g., in a CI secret or a private config store). They must never appear in commit history, PR diffs, or CI logs.

---

## Implications of 32k benchmark prompt

The benchmark prompt is 64× longer than the current `PREFILL_PROMPT_LENGTH = 512`. This has significant measurement consequences:

**Prefill latency** — `_measure_prefill_latency` runs two timed forward passes over the full prompt. At 32k tokens and ~0.048 s/tok, each pass takes ~26 minutes. This is impractical to run twice on every benchmark invocation.

Recommended changes:
- **Single timed prefill run** instead of two (warmup still runs once). The variance at this length is low enough that averaging adds little value.
- **Prefill measured as a side effect of the KV cache fill** that precedes the decode loop, rather than as a separate repeated forward pass. The 32k prompt is processed once; that wall time / 32768 is `prefill_spt`.

**Decode loop** — unchanged. After prefill fills the KV cache, `DECODE_LENGTH = 512` tokens are generated autoregressively as before.

**Peak RAM** — the 32k KV cache is substantially larger than the current 32-token seed. MLA cache at 32k context will dominate peak RAM for implementations that materialize it fully. This is intentional: it makes peak RAM a meaningful axis for long-context optimization, not just a function of model parameter size.

**Bandwidth** — 32k prefill reads the full model once in a batched pass. Decode bandwidth (per-token) is unchanged.

---

## What is NOT changed

- `DECODE_LENGTH = 512` (number of autoregressive tokens measured) — unchanged.
- `CORRECTNESS_EPSILON = 1e-2` — only relevant if hidden-state comparison is retained as an additional check.
- The `harness_hash` computation — prompts are external data, not harness code. Changing a prompt does not change the harness hash.
- Score formula — unchanged.

---

## Open questions

1. **Should we keep hidden-state comparison as a secondary check?** It catches quantization errors that happen to produce the same top-1 token but corrupt intermediate representations. Adds ~2× correctness check time. Recommend: keep as an optional `--strict-correctness` flag, off by default.
2. **Who validates prompt token lengths?** A `mlxfast validate-prompts` subcommand that tokenizes both local files and prints their lengths would prevent length-mismatch errors at run time.
3. **Prompt rotation policy.** Server prompts should be rotated each challenge round. Old prompts can be published post-round as historical reference.
