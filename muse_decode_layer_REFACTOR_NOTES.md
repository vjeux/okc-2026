# Refactor notes — `muse_decode_layer.py` → `muse_decode_layer_clean.py`

Board entry v157e. `prepare()` / `kernel_fn()` are the only entry points.

## Result

| | lines |
|---|---|
| Before: `muse_decode_layer.py` | 13,771 |
| After: `muse_decode_layer_clean.py` | 13,486 |
| Net | **−285 lines** |

No speedups claimed, nothing submitted, nothing run on GPU (none available
on this host; the host has no `torch` installed, so even an import check was
impossible here — see Verification).

## The headline finding: most of the file is live

The premise that "most of it is likely duplicated/dead" did **not** survive
measurement. Reachability analysis from `prepare()` / `kernel_fn()` (AST
Load-edges, plus the tuner / self-test paths they call) shows almost every
function is reachable:

- The three CUDA sources are each a *single* templated kernel whose `#if`
  branches are the per-build variants (`ATT7_CUDA_SRC` ≈ 4,846 lines,
  `MOE_CUDA_SRC` ≈ 3,503, the `bfp_*` template strings + generators ≈ 3,200).
  They look duplicated at a glance; they are not — one source, many
  compile-time builds. They are kept **byte-identical** (SHA-256 checked,
  see Verification).
- The `bfp_*` module looks like dead weight, but `bfp_prepare()` →
  `bfp__selftest()` → `bfp__autotune()` → `bfp_kernel_fn()` genuinely
  exercises most of it during `prepare()` / timed calls.
- The tuner (`_tune`) times every backend it can, so the FA3 fallback, the
  Triton split-K GEMMs, both boundary variants, Marlin, and all three MoE
  builds are live code even though only one wins per shape.

So the refactor is a *cleaning*, not a gutting: the live code stays, the
proven kernels are untouched, and only genuinely dead / duplicated /
misleading material was removed. Deleting more would mean deleting live
backends or tuned configs — a behaviour change, not a cleanup.

## What was removed (all dead, verified by AST reachability)

- **Broken dead helpers** — `_build` / `_module` referenced the undefined
  globals `CPP_SRC` / `CUDA_SRC`; `_inv_freq` referenced the undefined
  `_INV_FREQ`. None was ever called; each would have raised `NameError`.
- **Dead Triton MoE** — `_moe_align`, `_moe_w13`, `_moe_w2`, `_moe_combine`
  plus `_dequant` / `_e4m3_to_f32` (only used by `_dequant`). This is a
  whole earlier expert implementation that nothing launches; `_tune` only
  ever tries Marlin and the agent kernel. (`_f32_to_e4m3` stays: `_qk_prep`
  uses it.)
- **Dead bf16_proj generators** — `bfp__dispatch_src` (returned `""` with
  unreachable code after the `return`) and `bfp__mid_src` (its dispatch loop
  iterated `[]` and referenced the undefined `MID_SRC`; `_get_ext` never
  included a `mid` source). With them went their only consumers:
  `bfp_CONFIGS`, `bfp_MID_CONFIGS`, `bfp_MID_STAGES`, `bfp_MID_DU`,
  `bfp_D_PLAIN`, `bfp_D_COMP`, `bfp_PF`.
- **Dead `bfp_` TypedDicts** — `bfp_Weights` / `bfp_Config` / `bfp_Inputs`
  were the standalone bf16_proj entry's interface; this file passes plain
  dicts, so nothing referenced them. (The standalone
  `muse_bf16_proj_clean.py` keeps its TypedDicts — there they *are* the
  interface.)
- **`_prepack_frag`** — never called.
- **Dead constants** — `FORCE_MM` (never read), `bfp_MAX_SKINNY_M`,
  `bfp_TMA_KS`, `bfp_USE_TMA`, `bfp_SK2_ROT`, `_MA_JMAX` (only ever in a
  comment).
- **Disabled debug prints** — four `if False: print(...)` lines in
  `bfp__selftest` / `bfp__autotune` (they also carried the only `sys`
  references; `sys` and `math` imports are gone).
- **Dead tuner code** — `_tune`'s megakernel comparison: `tb` was hard-set
  to `1e30`, so `pb_ok[T]` was always `False` and `pbmax` always `0`.
  Replaced with a direct assignment (see Behaviour notes). The per-T log
  print was kept.
- **bf16_proj's standalone preamble** — its own `# okc:` header, long
  docstring, `from typing import TypedDict`, and mid-file
  `import torch / torch.nn.functional / sys / os as _os`, replaced by one
  section header; all imports now live at the top of the file.
- **A duplicated `LOG2E`** — defined at the top and again in the attention
  helpers.

## What was merged / renamed / de-duplicated

- **`_workspace7` / `_cluster_bufs7` were defined twice.** The first
  definition allocated, the second (at the end of the file) wrapped the
  first with `_guarded` via `_workspace7_raw` / `_cluster_bufs7_raw` alias
  rebinding. Now each is a single function that allocates guarded buffers
  directly — same cache keys, same sizes, same sentinel behaviour.
- **`_inv_freq` vs `_inv_freq7`**: two inverse-frequency helpers; the live
  one was `_inv_freq7`, the dead one `_inv_freq` (broken, see above). The
  live one is now `_inv_freq`, cache renamed `_INV_FREQ_CACHE`.
- **Expert fallback factored out**: `_after_attention`'s three-step
  Marlin↔agent fallback is now `_pick_expert_backend(...)`. The three
  steps are kept in order, not turned into a loop — a loop would hang when
  *neither* backend is available (the original terminates with `"agent"`).
- **Imports hoisted**: `threading` (`prepare`), `types` (`_marlin_bank`),
  `os`, `torch.nn.functional as F`, `load_inline` were imported mid-file or
  inside functions; all now at the top.

## Named constants (magic numbers removed from Python code)

New in the header: `QKV_ROWS = 2304` (was an unused `QKV`; the literal
`2304` in `_att_fits7` now uses it), `NUM_Q_HEADS = 16`,
`NUM_KV_HEADS = 2`, `HEAD_DIM = 64`, `Q_DIM = NUM_Q_HEADS * HEAD_DIM`
(1024, used for `q` / `q8` / `o_in` / `_decode_attention7`'s view),
`ROPE_THETA = 500000.0` (used in `prepare` and the attention self-test),
`ROUTER_MULT` now actually used in `_calib_bias` (it was defined but the
call site used the literal).

Deliberately **not** renamed: the `0.1767766952966369` literals inside the
`@triton.jit` kernels (`_route`, `_route_split`, `_moe_...` path) — a
module-global reference inside a Triton kernel changes what Triton
captures, so the literals stay there; and the literals inside the CUDA
sources, which are byte-identical by design.

## Comments

- New module docstring: the six-step pipeline, provenance, precision,
  selection, OKC rules, safety. It replaces a docstring that described the
  file as "built from four kernel sources" and listed per-entry provenance
  in passing.
- Timing-history comments condensed to WHY + conclusion (e.g. `PIN_PICK`:
  the isolated tuner misses PDL overlap, so these picks are pinned; the
  multi-line per-shape measurement table is gone). The verbose
  experiment-log comments inside the CUDA sources are untouched, because
  the CUDA is verbatim.
- The mangled `FORCE_GLUEV2` line (three comments concatenated on one line,
  the last two describing other knobs) is now one clean comment.

## Behaviour notes — what changed beyond provably-neutral cleanup

Everything above is provably neutral *except* these judgement calls, all of
which need **GPU gating** (a full judged run + the conformance suite) before
this file replaces the original:

1. **`_tune` pbmax**: `st["pbmax"]` was always `0` (see above) unless
   `FORCE_PBMAX` overrode it; the clean file assigns that directly. The only
   way this differs is if the dead code ever *didn't* produce `0`, which
   inspection rules out — but it is a behaviour-affecting line, so gate it.
2. **Merged `_workspace7` / `_cluster_bufs7`**: allocation sizes, cache
   keys, and sentinel placement are identical by construction, but the
   guarding now happens at allocation time rather than in a wrapper, so
   gate it.
3. **`_pick_expert_backend`**: verified exhaustively equivalent to the
   original three `if`s over
   `pick ∈ {marlin, agent, other, None} × bank × ma_ok` — still gate it,
   since it sits on the expert path.
4. **Everything else** is deletion of unreachable code, comment/whitespace
   changes, renames of private names, or literal→constant swaps with
   identical values. The CUDA sources, all launch configs, tile sizes, and
   tuner behaviour are unchanged.

Also kept deliberately, though they look dead: `_MA_ALT_SRCS` (the `@`-source
hook in `_ma_ext`, kept for parity with the standalone MoE entry),
`_MA_PDL`, `FORCE_PBMAX` / `FORCE_GLUEV2` (external override knobs),
`st["pbmax"]`, `state["tune_log"]`, and `bfp_GEMM_DIAG` / `bfp__diag`
(the disabled diagnostics hook). Removing them would change module/state
shape for no behaviour gain.

## Verification (no GPU available)

- `python3 -m py_compile` passes.
- All 14 CUDA source strings (`ATT7_CUDA_SRC`, `ATT7_CPP_SRC`,
  `MOE_CUDA_SRC`, `MOE_CPP_SRC`, `bfp_CUDA_SRC`, `bfp_MBAR_HELPERS`,
  `bfp_WG_HELPERS`, `bfp_TMA_SRC`, `bfp_MAP_SRC`, `bfp_GEMM_SRC`,
  `bfp_SK2_SRC`, `bfp_WIDE_SRC`, `bfp_GQ_SRC`, `bfp_CPP_SRC`) are
  SHA-256-identical to the original.
- Per-function AST comparison: of the functions present in both files,
  only 17 differ, and each differs only in the intended way (comment,
  constant, rename, deleted print, merged guard, factored fallback,
  pbmax). 12 functions were deleted (listed above); 1 added
  (`_pick_expert_backend`).
- Undefined-name scan: the original's genuinely undefined names
  (`CPP_SRC`, `CUDA_SRC`, `_INV_FREQ`, `MID_SRC`) are gone from the clean
  file. The remaining hits are checker false positives (except-bindings,
  closures, lazy `sglang` imports inside functions).
- Zero-reference top-level definitions in the clean file: only `prepare`
  (the harness entry point), as expected.
- `_pick_expert_backend` checked exhaustively against the original code
  (see above).
- The builder (`scratch-script`, not part of the submission)
  applied every deletion by AST span and asserted every replacement
  occurred exactly once, so a mismatch would fail loudly rather than
  silently produce a wrong file.

## Unsure about / needs the owner

- **`_plan7` / `_TUNED7` / the OKC_* attention constants** were left
  verbatim, timing comments included. They are the most "proud code"
  offenders left (long measurement logs), but their comments *are* the
  justification for tuned launch plans, and trimming them further risks
  losing the reason a plan value exists. Flagging in case you want them
  condensed too.
- **The bfp section still carries the `bfp_` prefix on ~60 names** and its
  own config-table style. Renaming them to match the rest of the file
  would diverge from the standalone `muse_bf16_proj_clean.py`, where the
  same names are kept — so the prefix was kept as the component's namespace.
- **No GPU gating was possible here.** In particular, the merged
  `_workspace7`, the pbmax change, and `_pick_expert_backend` should be
  exercised by a judged run before this file is used in place of the
  original, even though each is argued equivalent above.
- The builder script lives at `scratch-script` (scratch, not
  submitted). Re-running it reproduces the clean file from the original.
