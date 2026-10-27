# Refactor notes: `muse_prefill_layer.py` → `muse_prefill_layer_clean.py`

Board entry v116s. Refactor only: **no speedups claimed, nothing run on GPU**
(no GPU available in this session), nothing submitted.

## Result

| | Original | Clean |
|---|---|---|
| Total lines (`wc -l`) | 6836 | 6915 (+79) |
| Python lines (total minus the 5 verbatim C++/CUDA blocks) | 1201 | 1270 (+69) |
| Verbatim C++/CUDA block lines (content hash-identical; +10 is blank-line / `'''` boundary placement) | 5635 | 5645 (+10) |

The total grows because the clean file adds a ~60-line module docstring,
section headers, WHY comments, and the verbatim notice around the CUDA
strings, and wraps over-long lines. ~150 lines of dead Python were
removed; the rest is documentation and layout. The CUDA/C++ strings are
untouched (see below) — that is where most of the file's bulk lives.

`python3 -m py_compile` passes on the clean file.

## What the file is (module docstring, summarised)

One prefill layer step: QKV GEMM → Q/K RMS-norm + RoPE + FP8 quantisation
into the paged KV cache → attention (FA3, or the file's own Hopper kernel,
or an FP32 fallback on non-Hopper) → O-proj → residual-stream boundary →
router + top-8 on a side stream overlapped with the pooler GEMM → grouped
NVFP4 MoE (fused persistent kernel, SiLU\*up epilogue) → combine →
unpooler → second boundary. `prepare()` builds state once (extension,
repacked weights, tuned cuBLASLt / attention choices); `kernel_fn()`
allocates every activation fresh per call.

## Layout of the clean file

- 1–82: module docstring (algorithm + design rules), imports.
- 84–213: model geometry + **all tuning knobs, grouped** (MoE / boundary /
  router-combine / attention / cuBLASLt / self-test). Formerly these were
  scattered, several *after* their first use (the whole knob block at
  6570–6614, plus `LT_GEMMS`/`POOL_SM_RESERVE` mid-file).
- 215–330: Triton kernels (`_to_e4m3`, `_qk_prep_kernel`,
  `_norm_gate_kernel`, `_boundary_kernel`, `_combine_kernel`).
- 332–413: `prepare()` + `_residual` / `_split_router` / `_fast_scales_ok`.
- 415–528: dense GEMMs (`_lt_pin_index`, `_lt_rel_err`, `_tune_lt`,
  `_init_lt`, `_mm`).
- 530–582: repack (`_codes`, `_CODE_POS`, `_repack`).
- 584–823: attention (FA3, fallback, `_attention_norm_gate`,
  splits/class/variant, staging, `_attn_prep_split`, `_attn_custom`,
  synthetic cases, validation, tuning, `_choose_attention`).
- 825–883: MoE (`_max_tiles`, `_moe_forward`).
- 885–1011: `kernel_fn()` + `_run_qk_prep` / `_run_attention`.
- 1013–1235: self-tests (`_routes_from_counts`, `_time`, `_selftest`,
  `_selftest_attn2`, `_selftest_boundary`).
- 1237–6915: `_build` / `_build_lt` + the five C++/CUDA strings,
  **byte-identical to the original** (SHA-256, first 16 hex):
  - `CPP` 7fa15145be2c5b24, `GLUE` 9a156ab75e20d3f2,
    `_CUDA_SRC` b45ce8984ea14d0a, `LT_CPP` c44d27210d322210,
    `LT_GLUE` 6558a42f5178b013.

## What was removed (dead code)

Evidence for each is "no reference anywhere in the file, including
inside the CUDA strings' launch tables":

- `_e4m3_to_f32` — FP8→FP32 Triton helper, never called.
- `_boundary2_kernel` — Triton recompute-the-stream variant, never
  launched (only mentioned in a history comment, which is gone too).
- `_route_kernel` — Triton router/top-8, never launched; the CUDA
  `moe_route_kernel` does the job. (The name `moe_route_kernel` still
  appears in the verbatim CUDA — that is the live one.)
- `_selftest_attn` — FA3 timing probe, never called. `_selftest_gemm`
  — deep_gemm/quack import probe + GEMM timings, never called.
- `ATTN_VARIANTS[99/52/53]` — guarded by `"ATTN_NWG1" in
  _CUDA_SRC[:400]` etc. Those tokens are not in the first 400 chars of
  `_CUDA_SRC` (they live in `GLUE`), so the branches could never fire.
- `import re as _re` + the `ATTN_SMEM` regex — see "Provably neutral"
  below.
- Unused Python constants `N_EXPERTS`, `ROUTER_MULT` — they were
  defined but never read by Python; their values live in the verbatim
  CUDA. (Contrast `RMS_EPS` / `POST_NORM_EPS`, which were also unused
  but are now *used* by the Triton kernels — see below.)
- The single-iteration `for P1, P2 in ((8, 16),):` loop in `_selftest`
  — a leftover sweep shell; replaced by a direct assignment.
- Stale history in comments (dated tuning anecdotes, "v113 pinned the
  router by mistake", "TRICKS 5.16") — trimmed to the WHY that justifies
  each default. `MOE_CIT` / `CIT` still say the CUDA define must agree.

## What was deduplicated / renamed

- `_attn_variant` duplicated `_attn_class`'s whole if-chain; it is now
  `choice[_attn_class(cfg)]`. Mapping verified on the boundary cases
  (window / tokens > 4096 / batch > 1).
- `_attention_norm_gate` (FA3 + fused norm-gate epilogue) was inlined
  in three places (`kernel_fn`, `_attn_ref`, `_selftest_attn2.ref_fn`);
  it is one helper now. **Careful:** the original passed
  `q8.view(FP8)` in `kernel_fn` but raw `q8` in `_attn_ref` /
  `_selftest_attn2` — the helper takes the tensor as given, and
  `_run_attention` applies the `.view(FP8)` so each call site keeps
  its original dtype. This was caught in review of the rewrite, not by
  any test.
- `_lt_rel_err` — the cuBLASLt relative-error formula appeared twice
  in `_tune_lt`.
- `_max_tiles` — `T * TOP_K // 32 + 64` appeared in `_moe_forward`
  and `kernel_fn`.
- `_run_qk_prep` / `_run_attention` — the attention preamble of
  `kernel_fn` is now two named helpers; launch arguments unchanged.
- `_init_lt` (LT build + tune + failure print) and `_choose_attention`
  (attention validate/tune + failure print) extracted from `prepare`;
  `prepare` is now orchestration only, with the same order of
  side effects (LT → `synchronize` → attention → self-tests).
- `_attn_prep_old` → `_attn_prep_split`: "old" was a misnomer for a
  live split-staging path.
- `_NSM = [132]` (mutable list used as a module-global box) → `nsm`
  parameter on `_attn_splits`, passed from `st["nsm"]` at all three
  call sites. `_SIDE_STREAM = []` (list-box) → plain global
  `SIDE_STREAM`. `_FLUSH = []` (list-box) → `_FLUSH_BUF`.
  Same values, same laziness; the stream is still created once in
  `prepare` and kept out of the state dict (state holds only tensors
  and plain data).

## Named constants (no more magic numbers in the Python layer)

- Triton kernels spell shapes with `HQ`, `HKV`, `HD`, `Q_SIZE`,
  `KV_SIZE`, `QKV_SIZE`, `GATE_OFFSET` (1280), `RMS_EPS`,
  `POST_NORM_EPS`, `TOP_K`. Example: `_qk_prep_kernel`'s `2304` →
  `QKV_SIZE`, `1152` → `Q_SIZE + KV_SIZE`, `loc * 128` →
  `loc * KV_SIZE`.
- `ATTN_SPLIT_SLOT = HD + 2` (66: 64 unnormalised O + running max +
  running sum), `ATTN_SK_Q_BLOCK = 24` (stream-K q-rows per block),
  `ATTN_SK_MAX_PAIRS = 1536` (matches CUDA `ATTN_SK_MAXPAIRS`),
  `LOG2E`, `MOE_FINE_ORDER_SHIFT = 8` (bit 8 of plan/fused flags),
  `MIN_TILE_ROWS`, `MOE_TILE_SLACK`,
  `MAX_PADDED_ROWS_PER_EXPERT = 256`, `LT_ALGO_TOL`,
  `LT_MIN_SPEEDUP`, `LT_WORKSPACE_BYTES`, `MOE_SMEM_SM90/SM80/SKINNY`,
  `ATTN_SMEM_SM80_EXTRA`, `ATTN_VALIDATE_TOL`,
  `ATTN_Q_ROWS_PER_WARPGROUP`, `ATTN_KEY_TILE`, `ATTN_MAX_SPLITS`,
  `ATTN_SPLIT_PENALTY`, `FALLBACK_Q_BLOCK`,
  `SYNTH_SM_SCALE` / `SYNTH_QK_SCALE_FACTOR` (synthetic cases).
- `_to_e4m3` is the exception: kept byte-identical. Its constants
  *are* the E4M3 bit format (RNE bias `0x7FFFF`, subnormal threshold,
  saturation at code 126); naming them would obscure the bit trick,
  and its docstring now explains the trick instead.

## Provably neutral changes

- All of the above were checked by diffing each rewritten function
  against the original: every non-comment difference is one of the
  substitutions/extractions listed here.
- `ATTN_SMEM`: the original parsed `#define ATTN_ASTAGES` out of
  `_CUDA_SRC[:2000]` — but that define sits at byte 140570 of the
  string, so the regex *never* matched and the fallback `4` was always
  used. The clean file uses `ATTN_ASTAGES = 4` directly
  (`ATTN_SMEM = 83968`, unchanged). The coupling is now explicit:
  the constant must match the CUDA `#define`.
- `_attn_splits`: equivalence against a literal transcription of the
  original formula checked on 4,200 random
  (batch, q_len, window, page, variant, SM-count) cases — all agree.
- Pure-Python checks (module imported with stubbed torch/triton, no
  GPU): constants, `_max_tiles`, `_attn_class` / `_attn_variant`,
  `_attn_splits` — all pass.
- `COMBINE_PW_EARLY` stays `tl.constexpr(False)` — the Triton kernel
  needs a compile-time constant for its branch, and the original did
  the same.

## Needs GPU gating (not provable from source alone)

Nothing below is believed wrong; each is a change whose equivalence
depends on toolchain behaviour, so gate it on a GPU run before
treating the clean file as the submission:

1. **Triton kernels now reference module globals** (`HQ`, `HD`,
   `Q_SIZE`, `RMS_EPS`, …) instead of literals. Values are identical
   (asserted in the stub-import check), but Triton must capture them
   as compile-time constants and specialise the same way. Gate: run
   with `SELFTEST = True` — `_selftest_boundary` compiles
   `_boundary_kernel`, and flipping `QK_PREP_CUDA = False` once
   compiles `_qk_prep_kernel`; the Triton `_combine_kernel` is
   already on the `kernel_fn` path. Also run the fallback path's
   `_norm_gate_kernel` via `_selftest_attn2`.
2. **Extractions in the launch path** (`_run_qk_prep`,
   `_run_attention`, `_attention_norm_gate`, `_init_lt`,
   `_choose_attention`, `_max_tiles`). Argument order/values are
   preserved by diff, but confirm with `kernel_fn` outputs on the
   scored shapes (753 / 3012 / 16384, sliding + full) against the
   original, bit-for-bit or within the original's own tolerance.
3. **Global-box → plain-global / parameter changes** (`SIDE_STREAM`,
   `nsm` argument, `_FLUSH_BUF`). Same semantics, but stream
   wait-ordering is timing-sensitive: covered by the same
   `kernel_fn` comparison in (2).
4. **`ATTN_SMEM` / `ATTN_ASTAGES` coupling** — same value, but if
   the CUDA define ever changes, the Python constant must change
   with it (the original regex pretended to track it and did not).

Suggested gate: `SELFTEST = True` on Hopper + `kernel_fn` on the
scored shapes vs the original. No GPU was available here, so none of
this was run.

## Deliberately left alone (not a refactor's job)

- The five C++/CUDA strings are verbatim, including their
  `#ifdef` experiment branches (`MOE_DBG_*`, `ATTN_DBG_*`,
  `ATTN_NWG1`/`ATTN_V52`/`ATTN_V53`/`ATTN_PIPE_NORR`/`ATTN_RR_STEP_*`
  in `GLUE`, `MOE_CIT` paths, profiling hooks). Deleting them would
  be a CUDA change, not a cleanup. Same for the CUDA `boundary2`
  kernel + its `CPP` wrapper: dead (not in `_build`'s function
  list), but inside the verbatim strings.
- Cross-language literal duplication (e.g. the router scale
  `0.1767766952966369f` inside `moe_route_kernel` vs Python): the
  CUDA side is verbatim, so the Python side cannot be the single
  source of truth without touching CUDA. `ATTN_ASTAGES`,
  `ATTN_SK_MAXPAIRS`, and `ATTN_SMEM` are the three couplings called
  out with "must match" comments.
- The timed path still performs no host sync. The only syncs in the
  file are `prepare`-time tuning, self-tests, and the FP32 fallback
  attention (`.tolist()`), which is testing-only and was already
  there. `kernel_fn` still allocates every activation fresh per
  call; the only cross-call state is prepared weights, tuned
  choices, the compiled extension, and the side stream.
- Long lines inside the verbatim strings are untouched. Python lines
  were wrapped to ~110 columns, except a few `load_inline`
  call-site lines where wrapping would hurt the copy-paste match
  with the original.
