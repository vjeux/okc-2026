# Refactor notes — `muse_bf16_proj.py` → `muse_bf16_proj_clean.py`

Board entry v141. Goal: code to be proud of, behavior-identical.
No speedups claimed. No GPU run (none available yet) — see Verification.

## Line count

| | lines |
|---|---|
| Before (`muse_bf16_proj.py`) | 4111 |
| After (`muse_bf16_proj_clean.py`) | 4195 (+84) |

Net growth is documentation, not code: ~56 lines of dead code removed,
~140 lines added as the module docstring, a shared-constants section,
section banners, and four small helpers. The CUDA is byte-identical
(see Verification).

## What changed

### Module docstring (rewritten)
Explains the math, the lossless 12-bit packing + exact escape correction,
why the kernels stream weights once and split K the way they do, every
route and when it runs (`sk2` / `skinny` / `cgemv` / `sgemv` / `rgemv` /
`wide` / `gemm` / `gq` / `gqx` / `rk` / `cutlass` / `torch` / `torchT`),
and the file's design (one file, `load_inline` once, `prepare()` builds
layouts, `kernel_fn()` runs one projection, fresh outputs per call,
no host syncs in timed paths).

### Dead code removed (~56 lines)
- `_dispatch_src()` + `CONFIGS` + `D_PLAIN` + `D_COMP` + `PF`: the old
  LDG skinny dispatch. `_dispatch_src()` returned `""` immediately, its
  body was unreachable, and nothing called it.
- `_mid_src()` + `MID_CONFIGS` + `MID_STAGES` + `MID_DU`: the "mid" route
  was never built (`_get_ext` never called it), its config loop iterated
  over `[]`, and it referenced `MID_SRC`, which was never defined — calling
  it would have raised `NameError`. Unrecoverable dead code, not a route.
- `TMA_KS`, `USE_TMA`, `MAX_SKINNY_M`, `SK2_ROT`: constants never read
  anywhere.
- `TMA_TU`: a dict written by `_tma_srcs` but never read anywhere in the
  file (leftover bookkeeping for a multi-translation-unit split that never
  happened). The population loop was removed with it.

### Duplication removed
- Exponent-window selection (histogram → best 8-wide window → base/offset)
  was copy-pasted in 4 places (`pack_compressed`, `sk2_compress`,
  `cg_compress`, `sk2_balance_perm`). Now one helper, `_exponent_window`,
  plus `_escapes` for the outside-window test, used by all four.
- The skinny stream/counter/escape-table block was duplicated in
  `prepare()`'s `TMA_CONFIGS` and `TMA_ALT` loops. Now one helper,
  `_ensure_skinny_stream`, called by both (it is idempotent, as the
  original code was).
- `_run_gq` / `_run_gqx` were identical except config table, BN function,
  and entry-point name. Now one `_run_gq_family` plus two one-line
  wrappers. The entry point is selected with `getattr` on a literal
  string at each call site.

### Named constants (no magic numbers)
New "Shared constants" section: `EXPONENT_WINDOW`, `EXPONENT_VALUES`,
`MAX_ESCAPE_DENOMINATOR` (the `// 200` escape-share cutoff),
`BASE4_REPLICATE` (the `0x01010101`), `TMA_MIN/MAX_STAGES`,
`SM90/SM80_SMEM_BUDGET_BYTES`, `SK2_MIN/MAX_STAGES`,
`SK2_DEFAULT_RING_CAP_KB`, `SK2_RING_SMEM_BUDGET_BYTES`,
`SELFTEST_SEED` (1234), `AUTOTUNE_SEED` (4321), `AUTOTUNE_REPS` (24),
`L2_FLUSH_ROWS` (256 MiB sweep), `UNPOOLER_SLICE_ROWS` (the repeated
`1024` for the unpooler slice). `WEIGHT_ROWS` (was `w_rows`) and
`COMPRESSED_LAYERS` (was `COMPRESS`) renamed; `wgmma_defs` renamed
`wgmma_rs_defs` to pair with the existing `wgmma_ss_defs` (register-
vs shared-sourced A). Imports moved to the top (`threading`,
`importlib.util`, `os` — no more function-local imports).

### Structure / comments
- Section banners: CUDA sources, extension build, weight preparation,
  `prepare()`, routing/selftest/autotune, running one projection,
  entry point. Config tables stay next to the kernel they configure.
- Comments now explain WHY where it matters (e.g. why the uncompressed
  skinny path passes dummy/pmax/base4 `0`; why `prepare()` calls
  `_get_ext()` eagerly). WHAT-comments that just restated code were not
  added; existing WHY-comments in the CUDA were kept verbatim.

## Deliberately NOT changed
- **All CUDA/C++ source strings are byte-identical** (verified with `ast`:
  all 14 string constants, including `CUTLASS_SRC` with its NVIDIA license
  header). Same kernels, numerics, tile sizes, launch configs, dispatch
  tables. Nothing in a `r"""..."""` block was touched.
- **CUDA near-duplicates kept**: `rk_encode`/`rk_map` in `RK_SRC` duplicate
  `get_encode`/`make_map_2d` in `MAP_SRC`, but with different error-message
  prefixes (`"rk: ..."`), so merging them would change observable behaviour
  in the failure path. Same reasoning for the `rk_*` mbarrier helpers
  (different barrier scope) and the per-kernel reduction epilogues.
- **`build_patch_table` vs `sk2_table` kept separate**: same shape, different
  ordering semantics (patch table grouped by CTA; sk2 table k-ordered
  within each CTA because the escape warps walk K once).
- **`swizzle_tiles` + `DIAG_LAYERS` + the `GEMM_DIAG` block in `prepare()`
  kept**: currently unreachable (`DIAG_LAYERS` is `()`), but unlike
  `_mid_src` this is working diagnostic code gated by two flags — flipping
  `GEMM_DIAG` and populating `DIAG_LAYERS` re-enables it. Judgment call;
  deleting it would also have been defensible.
- **Config values untouched**: every table entry (`TMA_CONFIGS`,
  `SK2_CONFIGS`, `SK2_PICK`, `GEMM_CONFIGS`, `WIDE_CONFIGS`, `GQ_CONFIGS`,
  `GQX_CONFIGS`, `RK_SPLITS`, `CG_CONFIGS`, routes, split counts) is
  identical — verified by direct comparison.

## Verification (no GPU, no torch on this devserver)
- `python3 -m py_compile muse_bf16_proj_clean.py` — passes.
- `ast` comparison of all 14 CUDA/C++ string constants, orig vs clean —
  byte-identical.
- The 7 dispatch builders (`_tma_srcs`, `_gemm_src`, `_wide_src`,
  `_sk2_src`, `_gq_src`, `_gqx_src`, `_rk_src`) were executed from both
  files (pure-Python functions, extracted via `ast`) for `sm90=True/False`
  — assembled CUDA output byte-identical in all 14 cases. Same for
  `_tma_nstage` and `_sk2_geom` spot checks, and all config tables.
- Full `diff` reviewed hunk by hunk: every change is in Python code
  (docstring, constants, dead-code removal, helpers, renames, comments).
  Function inventory: only `_dispatch_src`, `_mid_src`, `wgmma_defs`
  removed; only `_exponent_window`, `_escapes`, `_run_gq_family`,
  `_ensure_skinny_stream`, `wgmma_rs_defs` added.

## Still needs GPU gating
Nothing in this refactor is *believed* to change behaviour, and the
checks above cover everything checkable without a GPU — but the file's
own safety net is the GPU path, so before this is submitted:
- Run on an H200: `prepare()` should pass `_selftest` with zero newly
  disabled routes, and `_autotune` should reproduce the same per-shape
  picks (the `[okc] tune …` log lines) as the original file on the same
  machine. Any difference in those logs is a red flag.
- The packing helpers (`pack_compressed`, `sk2_compress`, `cg_compress`,
  `sk2_balance_perm`) could not be executed here (no `torch` on the
  devserver); their refactor is a mechanical substitution of identical
  expressions, but a CPU-torch run comparing their outputs orig-vs-clean
  on random matrices would close that gap.

## Unsure about / judgment calls
- `TMA_TU` removal: no in-file reader exists, but if any external tooling
  imports `TMA_TU`, it will now fail. It was write-only bookkeeping.
- `_exponent_window` in `sk2_balance_perm` computes the clamped offset
  tensor and discards it (only `base` is needed there). Prepare-time only,
  negligible cost; kept for the single-helper shape.
- Kept `swizzle_tiles` / `GEMM_DIAG` scaffolding (see above) rather than
  deleting all unreachable code — flagging in case you prefer the stricter
  reading of "no dead code".
