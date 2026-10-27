# Refactor notes: `muse_attention_decode.py` → `muse_attention_decode_clean.py`

## What changed, in one paragraph

Same kernel, same numerics, same launch configs — with the tuning
archaeology removed. The ~140-entry `OKC_*` compile-time switch table had
grown a measurement log ("Oct 26 14:16 … 16.10 -> 15.50 us", "scout's v200
census", "v288 vf16 …"), and most of its branches were dead: the host build
passes only `-DOKC_SM90`, so every other switch is fixed to the value in
this file. This refactor evaluates that fixed table, deletes the dead
branches, drops the helpers that only the dead branches called, deletes the
defines nothing references anymore, and rewrites the surviving switch table
with WHY-only comments. The Python wrapper gets a module docstring, named
constants, and three small helpers. No algorithm, tile size, launch config,
or arithmetic was changed.

## Line counts

| Section | Before | After | Delta |
|---|---|---|---|
| Whole file (`wc -l`) | 5195 | 4293 | **−902 (−17%)** |
| Embedded CUDA body | 4846 | 3855 | −991 |
| Python header (docstring + constants + types) | 49 | 125 | +76 |
| `CPP_SRC` (the `run` declaration) | 9 | 9 | 0, verbatim |
| Python tail (host code) | 291 | 304 | +13 (helpers + guard) |

CUDA reduction breakdown: preprocessor fold −847 (dead branches + the
`#ifndef … #define … #endif` wrappers collapsing to one `#define` line
each), dead-function / unreferenced-define pass −128, macro-table comment
rewrite −16.

## Removed

**Dead preprocessor branches (−847 lines with their directives).** A small
evaluator walked the CUDA source with the fixed macro table (the only
unknowns: `OKC_SM90`, set by `_build` via `-D`, and `__CUDA_ARCH__`, set by
nvcc — groups conditioned on either were kept verbatim, including their
nested dead branches, which were still folded). This removed, among others:
every `#else` arm of the fifty-odd `OKC_* = 0/1` switches, the `#error
"OKC_DYN needs OKC_CONT"` (condition false), the `#ifdef OKC_SPINTEST` /
`#ifdef OKC_COOPGUARD_TEST` test hooks, the second `#ifndef OKC_PROF` /
`#ifndef OKC_PROFPV8` definitions, and the `#if OKC_XP_*` wrong-results
experiment arms (`XP_NORESC`, `XP_NOMASK`, `XP_MASKRO`, `XP_NOEXP`,
`XP_NOTR`, `XP_NOFENCE`).

**Dead device/host functions (11).** Helpers whose only call sites were in
folded-away branches, or which were never called at all: `tma_load_1d`,
`tma_row`, `mbar_init`, `mbar_wait` (the TMA path is off: `OKC_TMA = 0`),
`mma_fp8` (never called), `pt_map` (never called), `cp_async4z`
(never called after the fold), `ex2_e4m3x4_h` (only in the dead
`OKC_EXH8` arm), `ex2_poly` (only in the dead `OKC_EXPPOLY` arm),
`wgmma_m64n64k16_f16_tb` (only in the dead `OKC_PVW` arm). Each was
verified to have exactly one remaining reference — its own definition —
before removal. Their dedicated `//` comment lines went with them.

**Unreferenced defines (22).** `OKC_NOCOMPUTE`, `OKC_PFAST`, `OKC_EXH2`,
`OKC_PROBE_O8`, `OKC_P8EMU`, `OKC_BRFREE`, `OKC_PREUNR`, `OKC_FIN1`,
`OKC_CLR2`, `OKC_DYNK`, `OKC_TANHSIG`, `OKC_XP_NORESC`,
`OKC_XP_MASKRO`, `OKC_DSHFL`, `OKC_YIELD`, `OKC_PACK4`, `OKC_FASTSC`,
`OKC_LAZY1`, `OKC_QROLL`, `OKC_CLR4`, `OKC_SMDBG`, `OKC_PROFDEP` — each
referenced nowhere in the folded source (their only uses were in folded
directives). Removed to fixpoint.

**Dead Python.** A `while cs * ncl > tiles and ncl > 1` loop in `_plan`
(`ncl` is 1 at that point, so the guard is false and the loop never runs);
`import math` (unused); a duplicated `_TUNED` definition introduced during
assembly (caught by review); the comments in `_plan` / `_launch` that were
pure measurement logs (dates, timings, "v101/v288"). The surviving tuning
values are unchanged — only their logs are gone.

## Deduplicated (2)

* `row_map` duplicated `get_encode()`'s driver-entry-point lookup verbatim
  (its own `static EncodeTiledFn` + the same `cudaGetDriverEntryPoint` /
  `TORCH_CHECK`). It now calls `get_encode()`. Same function pointer, same
  check, one static instead of two; the only copy of that lookup left is
  `get_encode` itself.
* Python: `_inv_freq` / `_num_sms` shared the dict-cache pattern → one
  `_cached(cache, key, make)` helper. `_rope` (fallback only) now reuses
  `_inv_freq(theta, device, d // 2)` instead of recomputing the same
  frequencies inline. `_plan` / `_launch` share the repeated
  `max(1, min(...))` clamps → `_splits(tiles, quota)` and `_copies(n, cap)`.

## Two deliberate (non-mechanical) changes

1. **`_build` rejects `-DOKC_*` flags.** Previously `_build(tag, flags)`
   forwarded extra nvcc flags, so a caller could pass `-DOKC_PROF=1` and
   silently rebuild a different kernel than the table in this file
   describes. The table is now documented as a closed world
   (only `-DOKC_SM90` comes from outside), and `_build` raises `ValueError`
   on any `-DOKC_*` flag, before importing `load_inline`. The default
   build (`_build()`, the only caller) is unaffected.
2. **The `w0` bitfield is named.** `_launch` packed `tma` as
   `… | (coop << 1) | (min(max(w0, 0), 4095) << 2) | (… << 16)`; it now
   uses `COOP_BIT`, `W0_SHIFT`, `W0_MAX`, `PDL_BIT` with a comment giving
   the layout (bit 0 = TMA from `plan["tma"]`). Same bits, same values.

Everything else is mechanical (fold, delete, rename-to-constant, recomment).

## Verification (no GPU was available; nothing was run on GPU)

* `python3 -m py_compile` passes on the clean file.
* **Preprocessor oracle.** `gcc -E` on the original vs. the clean CUDA
  under four configurations (`OKC_SM90` 0/1 × `__CUDA_ARCH__`
  undefined/900/800): the preprocessed outputs are byte-identical except
  exactly the removed dead functions, the removed `pt_map`, and the one
  `row_map` line (`const EncodeTiledFn encode = get_encode();`). This
  proves the fold and the dead-code removal change no compiled code.
* **Host-logic sweeps with a stub `torch`.** `_plan`: 124,800
  shape/arch/SM-count cases × flips of every launch switch
  (`OKC_F2/S2/FLAT1B/PLAIN8/FLATCOOP/PDL/W0_*`) — 0 mismatches.
  `_launch`: 6,912 cases (shapes × overrides × `pdl`) comparing the full
  argument list passed to `run()` — 0 mismatches. `_supported`: 0
  mismatches over its config grid.
* **CUDA text.** The clean file's `CUDA_SRC` is exactly the folded +
  cleaned CUDA; `CPP_SRC` is byte-identical to the original. The
  `okc_attn_v298` build name is unchanged.

## What was deliberately NOT done (unsure / out of scope)

* **Runtime `if (OKC_*)` / `if constexpr` checks were not folded.** The
  source still contains constant-valued checks such as
  `if (OKC_XP_NOEXP)`, `if constexpr (!OKC_XP_NOTR)`, and
  `if (OKC_SPINDBG && …)`. ptxas folds them, so removing them would not
  change the compiled kernel — but doing that safely needs a real C++
  parser, not regexes, across 3,800 lines. Left for a follow-up, with GPU
  gating, since a mis-parse would be silent.
* **`segment()` was not split.** At ~2,300 lines it is the one function
  that is still too big, but it is one logical unit (the per-segment main
  loop) and its branches share dozens of locals; splitting it is a
  behavior-risk change, not a cleanup. Same for the long flat-loop bodies.
* **Magic numbers inside kernel bodies were not renamed wholesale.** The
  Python side is fully named now; inside CUDA, tile-geometry literals
  (`64`, `32`, `128`, `2048`, …) are hardware geometry repeated in
  patterns a mechanical rename could easily get wrong (e.g. `64` is
  `HEAD_DIM`, `TILE_KEYS`, a byte count, and a thread count in different
  places). A rename pass belongs with a `nvcc` + SASS-diff gate, not with
  `gcc -E`.
* **Body comments still carry a few tuning parentheticals** ("measured
  slower", "scout's v200 census") where they are the actual reason a
  variant is off. Only the macro table and the Python side were fully
  de-historied, plus the single dated fragment that was purely a log.
* **No speedups are claimed.** No GPU was available, nothing was timed,
  and the build name / configs are unchanged, so there is no performance
  claim to make. GPU-gate before submitting: compile + run the OKC
  judge's checks on the clean file (the equivalence evidence above is
  strong, but it is not a GPU run).
