# Refactor notes — `muse_moe_w4a16.py` (Board entry v216x) → `muse_moe_w4a16_clean.py`

## Rule of the refactor

Behavior-identical, verified mechanically where possible without a GPU.
The CUDA kernel bodies are **byte-identical** to v216x except for the
dead-code deletions listed below and three comment-only insertions
(a roadmap header and two notes on the duplicated `skinny_ring`).
The Python functions are identical after substituting the new named
constants for their literals (checked with an AST-level function diff),
except for the two factored packer helpers and added docstrings/comments.
No algorithm, tile size, launch config, flag tuple, or numeric formula
was changed. Nothing was claimed about speed: no GPU was available,
nothing was timed, and no speedup is claimed.

## Line counts

| Section | Before | After | Delta |
|---|---|---|---|
| Total | 4326 | 4439 | +113 |
| Prologue: docstring + imports + TypedDicts + constants | 43 | 188 | +145 |
| `CUDA_SRC` body | 3501 | 3453 | −48 |
| `BW_SRC` body | 336 | 336 | 0 |
| Python after `CPP_SRC` | 437 | 453 | +16 |

The file got longer because the missing documentation now exists: a
~70-line module docstring (algorithm, scheduling, variant selection,
OKC constraints) plus named constants with WHY comments. Code got
shorter: the CUDA source lost 48 lines net (dead code out, a few
comment lines in).

## What the mess was (catalogue)

**Dead code — removed (11 items).** Each was verified dead by
whole-file reference counting (definition only, zero call sites):

* CUDA helpers: `dq_hi`, `chunk_mma`, `sk_load_w`, `sk_stage_x`,
  `wg_pin`. The skinny helpers were superseded by the TMA ring path
  (`ring_issue` / `load_x_regs`); `chunk_mma` by the inlined dequant
  inside the rings; `wg_pin` (an empty register-pinning asm stub) was
  never wired in.
* Compile-time knobs defined but never referenced: `OKC_FNST`,
  `OKC_FPF`, `OKC_FCOST2`, `OKC_FCOSTB2`, `OKC_WGSREL32`,
  `OKC_WGSPROF`. Note: `_ext` still passes `-DOKC_FNST=` on its nvcc
  command line and the fallback tuple still carries `-DOKC_FPF=5`;
  both were already no-ops (the macros were never read), and both are
  kept so the compile command is unchanged.

**Duplication — kept deliberately, documented, not merged:**

* `skinny_ring` exists twice (`#if SKDYN` / `#else`, ~340 lines each).
  The SKDYN variant claims tail units from a run-time atomic pool; the
  static variant issues a fixed range. The pool-claim logic interleaves
  with the ring refill, so a merge would risk changing scheduling.
  A comment at the `#if` now says this; the bodies were left alone.
* `WG<1..8>` and `WGSMMA<1..8>` are exhaustive wgmma-shape
  specializations (one per n-tile count). Repetitive by nature; each
  body is a distinct instruction shape, so they stay.
* The two skinny-epilogue blocks inside the two `skinny_ring`s and the
  WGS/WS epilogue variants are the same situation: near-duplicates
  whose divergence is load-bearing scheduling, not copy-paste.

**Magic numbers.** The Python half used bare `4096 / 32 / 8 / 256 / 384`
for five different things (model dim, expert count, routes per token,
local experts, chunk bound). Those are now named constants (below).
The CUDA half still contains ~4,400 integer literals: most are bit
layout, swizzle, and mbarrier arithmetic where a name would have to
carry the same number anyway (e.g. `0x81C081C0` in `dq4`, `1024 >> 4`
in `wg_desc_sw128`). Extracting them would churn every kernel body
for no behavior or readability gain, so they were left — the
load-bearing ones already had `constexpr` names (`HD`, `RB_A`,
`UNIT_BYTES`, …) or WHY comments.

**Unclear sections — clarified without touching code:**

* The module docstring was 6 lines and described only the kernel
  split; it now explains the algorithm (routing → Phase A GEMM1+SwiGLU
  → Phase B GEMM2+norm stats → final pass), the persistent-CTA /
  stream-K scheduling, the four compile-time variants, and the
  OKC constraints.
* `prepare`'s `state["w"]` list (14 tensors, order defined only by the
  C++ launcher's indices) now has a comment mapping each slot.
* `_make_scratch` now documents the three scratch buffers' layouts as
  the launcher indexes them.
* The CUDA source gained a roadmap comment mapping the section order
  (`route` → skinny ring → fat stream → WS → WGS → `run_phase` →
  `moe_fused` → `okc_run`).

**Commented-out experiments:** none found. There are no `#if 0`
blocks and no commented-out code in either file; the `// (Oct 26: …)`
trailing notes are tuning history on live knobs, kept. The one
out-of-order item noticed — `bw_run`'s `case 13` sits after `case 24`
in the switch — is cosmetic only (switch order is irrelevant) and
was left alone rather than churn the diagnostics file.

## What changed (restructure)

* **Prologue (new shape):** module docstring → imports → `TypedDict`s
  → one constants block (model shape, chunking, WGS scratch, variant
  define tuples + thresholds, feature flags, diagnostics flags).
  The variant tuples (`MAIN_DEFINES`, `TINY_DEFINES`, `TINY2_DEFINES`,
  `MID_DEFINES`, `MAIN_FALLBACK_DEFINES`) moved up out of the middle
  of the Python code so the dispatch table reads as a table; their
  contents are byte-identical.
* **Named constants (Python):** `HIDDEN=4096`, `TOP_K=8`,
  `N_EXPERTS_TOTAL=256`, `LOCAL_START=0`, `LOCAL_EXPERTS=32`,
  `SHARDING="ep8"`, `CHUNK_TOKENS=384` (was `MMAX` — renamed because
  `MMAX` also exists as an unrelated CUDA `constexpr`; the CUDA one is
  untouched), `EMAX`, `JMAX`, `WGS_SLOTS`, `WGS_FIXW`, `WGS_FIXF`,
  `WGS_XROWS`, `MAIN_NS`.
* **Factored packer helpers (the only Python logic refactor):**
  `_split_nibbles` (the nibble-split prologue shared by `_pack_codes`
  and `_pack_codes_nat`) and `_scatter_code_bits` (the 4-bit scatter
  loop both packers ran inline). Same tensor ops in the same order;
  callers assign the returned tensor (`word = _scatter_code_bits(...)`)
  rather than relying on in-place `|=` mutation.
* **`_ext`, `prepare`, `kernel_fn`, diagnostics:** bodies unchanged
  apart from docstrings, constant substitution, and comment moves
  (e.g. the extension-cache comment now states it caches compiled
  code, never data; `kernel_fn`'s docstring states the fresh-output /
  no-host-sync contract).
* **CUDA:** only the deletions above + the roadmap / `skinny_ring`
  comments. Function order, `#if` branches, and all bodies untouched.
* **`BW_SRC` / `CPP_SRC`:** byte-identical.

## What was deliberately NOT changed

* All `-D` flag tuples, `OKC_*` defaults that are actually read, and
  the `#if`/`#else` variant branches — these are tuned behavior.
* The `skinny_ring` duplication and the `WG`/`WGSMMA` specializations
  (see above).
* CUDA integer literals (see above).
* The diagnostics (`_diag`, `_diag_bw`, `_gen_input`, `_time_graph`):
  dev tooling, gated by `DIAG=False`. The `28.31` in `_diag` is now
  commented as approx. MB streamed per touched expert (see
  "Unsure" below).
* `ALT_SRCS` / `@src` source swapping in `_ext`: an extension point,
  empty in a submission.

## OKC rules — compliance audit

* One Python file; native CUDA only via `load_inline` (`_ext` for the
  main source, `_diag_bw` for the diagnostics-only `BW_SRC`). No other
  extension mechanism, no subprocess compilation.
* CUDA-graph capturable: `kernel_fn` contains no `synchronize`,
  `.item()`, `.cpu()`, or data-dependent host control flow. Host syncs
  in the file: one in `prepare` (one-time setup, after compiling) and
  five inside the `DIAG`-gated diagnostics (`_time_graph`, `_diag_bw`,
  `_diag`). `kernel_fn` allocates a fresh `out` every call and never
  reads a tensor back.
* Fresh inputs each call, no caching across calls: the only
  module-level cache is `_EXTS` (define tuple → compiled extension
  module). Weights/scratch are built once in `prepare` and reused as
  the task's API requires; no activations or outputs are cached.

## Verification performed (no GPU, no torch on this host)

* `python3 -m py_compile` passes on the clean file.
* CUDA diff: after stripping the three inserted comment blocks, the
  `CUDA_SRC` diff vs. the original is **pure deletions: 61 lines
  removed, 0 added** — the 11 dead items plus the orphaned
  skinny-unit section header their removal left behind.
* `BW_SRC` and `CPP_SRC` are byte-identical to the original.
* AST-level function diff (Python, after constant substitution):
  every function is identical to the original except docstrings /
  comments and the two factored packer helpers; no function was added
  or removed besides `_split_nibbles` / `_scatter_code_bits`.
* Sync audit (grep): the only `synchronize` calls are the setup one
  in `prepare` and the diagnostics ones above.

## NOT verified — needs GPU gating before this replaces v216x

* **The file was never compiled, run, or timed.** This host has no
  GPU and no torch installed, so `load_inline` never ran, the packers'
  outputs were not executed (their equivalence rests on the AST/op-order
  argument above), and the kernels' numerics were not compared.
  Gate on GPU: build all four variants, run the task's correctness
  check at several batch sizes (including the variant boundaries
  M = 4 / 8 / 16 and the chunk boundary M = 384 / 385), and run the
  precision check with `OKC_PRMTSC` both on and off.
* The dead-code deletions are provably neutral by reference counting
  (zero call sites / zero references), but a GPU build is the real
  proof that nothing else (e.g. an external tuning script passing
  `-DOKC_FNST=…`) depended on them. Nothing in this repo references
  them; an out-of-tree script was not checked.

## Unsure about (flagged, not guessed into the code)

* **`28.31` in `_diag`:** commented as approx. MB per touched expert
  (NVFP4 codes + scales for gate/up + down: ~16.8 + 2.1 + 8.4 + 1.0 MB
  by my arithmetic). Plausible, diagnostic-only, but not confirmed
  against the original author's intent — verify against the tuning
  history before trusting the printed TB/s.
* **CUDA/Python constant mirroring:** `HIDDEN` (Python) vs `HD`/`ID`
  (CUDA), `WGS_FIXF`/`WGS_XROWS` (both sides), `CHUNK_TOKENS` vs
  CUDA `MMAX` are duplicated by hand across the `load_inline`
  boundary — there is no mechanism to share them. If one side ever
  changes, the other must change with it; the prologue comment says
  this.
* **`_make_scratch` semantics:** the docstring claims scratch[0]'s tail
  is `ss1` because the launcher takes `ss1 = ib + 8 + 1024`. That
  matches `okc_run` (`p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024)`)
  and the `TORCH_CHECK` on `scratch[0]`'s size. Stated from the code,
  not from running it.
* **Whether `MMAX` → `CHUNK_TOKENS` renames anything else:** checked —
  `MMAX` appears in the clean file only as the CUDA `constexpr`
  (untouched) and in no other Python identifier.
