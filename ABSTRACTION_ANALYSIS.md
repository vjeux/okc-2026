# Abstraction analysis: the 5 refactored OKC submissions

**Question:** "We have ~5k lines for all of them — are we sure everything is needed
and there's no abstraction needed?"

**Answer, in one paragraph:** The premise is off by ~6.6× — the five `*_clean.py`
files total **33,328 lines**, not ~5k. Of that, **~11,300 lines are a forced
copy**: `muse_decode_layer_clean.py` re-embeds the attention, MoE and bf16
submissions wholesale because each OKC submission must be one self-contained
file, so the aggregate double-counts them. The remaining **~22,000 unique
lines** are mostly *not* duplication: the variant explosion is
*instantiation* (one template compiled many times with different
`-D` flags / config-table entries), not copied code, and the kernels that
look alike implement genuinely different data paths / schedules. The
genuinely mergeable duplication I could prove from the text is **~700 lines**
(plus the ~11,300-line structural copy in decode, which is only removable
if the one-file rule changes). No speedups are claimed anywhere here; no
GPU was available, nothing was run, no clean file was edited.

**Method (CPU only):** `wc -l`, segment spans from the triple-quote
boundaries, `grep` inventories, and `diff` of the corresponding sections
across files (content-only ranges, and `diff -I` ignoring `#`/`//`/blank
where noted). Every claim below cites `file:lines`. The `*_REFACTOR_NOTES.md`
files were used as secondary sources only where they state a verified
equivalence (e.g. SHA-256-identical CUDA strings); the counts are mine.
Line ranges are inclusive, 1-based, from the clean files as they stand.

---

## Top 5 abstraction opportunities (ranked by lines saved × confidence)

| # | Opportunity | Lines saved (est.) | Confidence | Why / gate |
|---|---|---|---|---|
| 1 | **Decode's re-embedded copies** — `muse_decode_layer_clean.py` contains ATT7 + MOE + bfp verbatim; generate decode from the three standalone files instead of hand-maintaining copies | ~11,300 (10,471 CUDA/C++ strings + ~900 Python packers/wrappers) | Counting: high. Actionability: **low under the current one-file rule** — it is a rule change / generation step, not a code edit | Copies have already diverged (see §2A); sync + GPU-gate all three components |
| 2 | **Generate the hand-expanded wgmma PTX specializations** — MoE `WG<1..8>` + `WGSMMA<1..8>`, prefill `wgmma_rs<32..256>`, attn e4m3 wgmma wrappers → Python generators like bf16 already has | ~290 (MoE ~188 + prefill ~60 + attn ~40) | **High** — the generator pattern already exists in-repo and is proven | Byte-identical generated CUDA (`diff` of assembled source) + `ptxas` SASS diff; then per-shape timing |
| 3 | **One `skinny_ring` template** in MoE — SKDYN (dynamic pool) + static variants are two ~340-line functions whose diff is 185 lines; factor the pool-claim into a `bool DYN` branch / policy | ~200–250 | Medium — the epilogue 3-way block is near-identical in both; the pool-claim interleaves with refill, which is the risk | Ablation: build `DYN=0/1` from one template, diff SASS vs both originals, time M at the SKDYN boundary (M = 4/8/16, chunk 384/385) |
| 4 | **Merge `attn_prep` / `attn_prep2`** (prefill) — two ~100-line kernels, ~60% shared (PRMT V-transpose identical); `attn_prep` is already `template<int HALVES>`, `attn_prep2` is the hardcoded 2-head version | ~50 | Medium-high — grid mapping + smem layout differ; the rest is shared text | Ablation: one `template<int HALVES, bool SPLIT>` build, correctness vs FA3 reference on the scored shapes (753/3012/16384, sliding + full), time the prep kernel standalone |
| 5 | **One layout-parameterized packer** per family — MoE `_pack_codes` vs `_pack_codes_nat` (differ in 4 lines); bf16 `pack_compressed` / `sk2_compress` / `cg_compress` (same exponent-window + escape core, different fragment order) | ~65 (MoE ~15 + bf16 ~50) | Medium — MoE pair is mechanical; bf16 trio's layout *is* the interface, so the saving is the shared prologue/epilogue only | CPU-torch equivalence on random matrices (packed bytes + escape tables byte-identical) + `prepare()` selftest on GPU |

Everything else I examined is either already templated, genuinely different,
or saves so little it is not worth the behaviour risk (§2, §5). The
~700-line actionable total (rows 2–5) is ~2% of the 33,328 total and ~3% of
the ~22,000 unique lines — that is the honest scale of "abstraction needed"
inside the current rules.

---

## 1. Line-count breakdown

### 1.1 Totals — 33,328, not ~5k

| File | Total | CUDA/C++ string spans | Python (rest) | Role |
|---|---|---|---|---|
| `muse_attention_decode_clean.py` | 4,293 | 3,863 | 430 | 1 fused attention kernel |
| `muse_bf16_proj_clean.py` | 4,195 | 2,751 | 1,444 | Dense bf16 projections, 13 routes |
| `muse_moe_w4a16_clean.py` | 4,439 | 3,799 | 640 | NVFP4 MoE, 1 persistent kernel, 4 builds |
| `muse_prefill_layer_clean.py` | 6,915 | 5,635 | 1,280 | 1 prefill layer: QKV → attn → MoE → combine |
| `muse_decode_layer_clean.py` | 13,486 | 10,471 | 3,015 | 1 decode layer: embeds attn + MoE + bf16 + Triton glue |
| **Total** | **33,328** | **26,519** | **6,809** | |

"CUDA/C++ string spans" = the lines between the `NAME = r"""` / `r'''`
assignment and its closing quote, delimiters included (spans verified
against the lone-`"""`/`'''` lines; e.g. attn `CUDA_SRC` 126–3979,
`CPP_SRC` 3982–3990; MoE `CUDA_SRC` 189–3643, `BW_SRC` 3645–3982,
`CPP_SRC` 3984–3989; prefill `CPP` 1271–1420, `GLUE` 1422–1691,
`_CUDA_SRC` 1693–6831, `LT_CPP` 6833–6840, `LT_GLUE` 6842–6909; decode
`ATT7_CUDA_SRC` 107–4952, `ATT7_CPP_SRC` 4953–4961, `MOE_CUDA_SRC`
5117–8619, `MOE_CPP_SRC` 8620–8625, `bfp_*` strings 8890–11162).

### 1.2 Inside the CUDA/C++ spans (lines)

| File | Blank | `//` comment | `#` directive | Other (code) |
|---|---|---|---|---|
| attention_decode | 54 | 303 | 379 | 3,127 |
| bf16_proj | 73 | 149 | 257 | 2,272 |
| moe_w4a16 | 61 | 197 | 788 | 2,753 |
| prefill_layer | 112 | 326 | 966 | 4,231 |
| decode_layer | 161 | 657 | 1,868 | 7,785 |
| **Total** | **461** | **1,632** | **4,258** | **19,168** |

Two readings: (a) only ~19k of 33k lines are CUDA *code* — the rest is
Python host code (~6.8k), preprocessor directives (~4.3k, mostly the
`#ifndef OKC_*` / `#if` variant scaffolding), comments (~1.6k `//` in
CUDA plus the Python `#`/docstring comments), and blanks. (b) The
`//` comments are disproportionately WHY-comments on tuned choices
(e.g. attn `muse_attention_decode_clean.py:1282-1330`, MoE
`muse_moe_w4a16_clean.py:918-924`); deleting them "saves" lines but
deletes the only record of why a variant is off. They are not dead
weight — see §5.

### 1.3 Per-file kernel / variant inventory

**Attention decode** (`muse_attention_decode_clean.py`) — *one* kernel,
already templated. `Cfg<L, FLAT>` (`:268-300`) computes thread counts,
stages, register split from `L ∈ {1, 8}` and `FLAT`. `segment<L, FLAT,
ROLE, MODE>` (`:1282-~3215`, ~1,930 lines) is the single per-segment
main loop; `attn_kernel<L, FLAT, MODE>` (`:3336-~3978`) is the single
`__global__` entry (the file's only `__global__`, `grep __global__` = 1).
The "variants" are *instantiations*: `launch<1>` / `launch<8>`
(`:3976-3977`), a 4-entry `KernFn` table of `MODE` builds
(`:3792-3796`), and `Cfg` booleans (`S2`, `F2`, `WS`, `ONE`). There are
118 `#define OKC_*` lines, but the refactor notes establish the host
passes only `-DOKC_SM90` — the rest are fixed compile-time constants,
not runtime branches. The 35 `#if/#ifdef/#ifndef` lines are the
compile-time selection of those constants.

**MoE** (`muse_moe_w4a16_clean.py`) — *one* `__global__` production
kernel `moe_fused` (`:3124`, plus 6 `BW_SRC` microbenchmark kernels at
`:3661-3845`, diagnostics only). Variant selection is *four
`-D`-tuples over the same source* (`MAIN/TINY/TINY2/MID_DEFINES`,
`:125-148`, picked by M in `kernel_fn`, `:4273-4280`) — not four
kernels. Inside: `route()` (`:737` ff.), `skinny_ring` ×2 (`:925-1268`
SKDYN, 345 lines; `:1271-1610` static, 340 lines), `fat_stream` /
`fat_compute` / `fat_compute_wg` (`:1721-2093`), WGS producer/consumer
(`:2375-3064`), `run_phase` (`:3065`), `WG<1..8>` (`:1612-1719`, 108
lines), `WGSMMA<1..8>` (`:2490-2604`, 115 lines), epilogues `epi_A` /
`epi_B` (`:737-760`). 248 `#if` lines, 72 `#define OKC`.

**bf16_proj** (`muse_bf16_proj_clean.py`) — 10 `__global__` kernels in
7 template families + 2 Python generators: `skinny_tma` (`TMA_SRC`,
`:336-802`, 467 lines), `gemm90` (`GEMM_SRC`, `:868-1046`),
`sk2` (`SK2_SRC`, `:1049-1566`, 518), `wide` (`WIDE_SRC`, `:1569-1929`,
361), `gq` (`GQ_SRC`, `:2083-2325`), `gqx` (`GQX_SRC`, `:2333-2539`),
`rgemv`/`cgemv`/`sgemv` (`RGEMV_SRC`, `:2578-2842`, 3 GEMV variants),
`rk` (`RK_SRC`, `:2891-3071`), CUTLASS (`:3122-3207`). Each family is
*one* templated kernel; the multiplicity is in the Python-generated
`@@*_DISPATCH@@` tables (`:2040, 2080, 2556, 2853, 2888, 3484`), which
expand config tables (`TMA_CONFIGS`, `GEMM_CONFIGS`, `SK2_CONFIGS`,
`WIDE_CONFIGS`, `GQ_CONFIGS`, `GQX_CONFIGS`) into `if (mt == …) launch<…>`
lines. 13 routes are documented in the module docstring (`:2-69`):
`sk2`/`skinny`/`cgemv`/`sgemv`/`rgemv`/`wide`/`gemm`/`gq`/`gqx`/`rk`/
`cutlass`/`torch`/`torchT`.

**Prefill** (`muse_prefill_layer_clean.py`) — 19 `__global__` kernels,
but in 3 already-templated families plus small singletons: MoE GEMM
`moe_gemm_kernel` (`:2175`, 191 lines) / `moe_gemm_persistent` (`:2781`,
165) / `moe_fused_persistent` (`:3193`, 414) sharing
`consumer_tile` (`:2073`) / `consumer_item` (`:2364`) /
`consumer_item_ni` (`:2945`); attention `attn_fwd_kernel` (`:6126`,
345 lines, template `<NWG, EXPMODE, OVL, PIPE, PIPE2, RRS, HALF,
PIPE2H>`) / `attn_fwd_sk_kernel` (`:6499`, 178) / `attn_combine*`
(`:6678, 6743, 6773`); `attn_prep` (`:4460`, 101, `template<int
HALVES>`) / `attn_prep2` (`:4561`, 110); `moe_plan` / `moe_gather` /
`moe_route` / `moe_combine` / `boundary` / `boundary2` / `qk_prep` /
`kv_gather`. Attention variants are 7 entries in `ATTN_VARIANTS`
(`:169`) mapped to `attn_fwd_kernel<NWG, EXPMODE, OVL>` instantiations
in the `GLUE` launch switch (`:1619-1644`); the `#ifdef ATTN_NWG1 /
ATTN_V52 / ATTN_V53 / ATTN_PIPE_NORR` blocks (`:1626-1641`,
`:1674-1688`) are experiment arms of that same switch. 289 `#if`
lines. Python (1,280 lines) is host orchestration + 5 Triton kernels +
`prepare`/`kernel_fn`/selftests.

**Decode** (`muse_decode_layer_clean.py`) — a composite. Its CUDA is
exactly three embedded submissions (see §2A) plus *no* new CUDA kernel.
Its unique Python is the Triton layer (`_gemm_splitk`, `_gemm_splitk_ng`,
`_normgate`, `_qk_prep`, `_attn_split`, `_attn_combine`, `_gemm_o`,
`_bnd1_apply`, `_route_a/b`, `_bnd2_*`, `_gate_ab`, `_boundary1`,
`_route`, `_route_split`, `_pexp_norm_sum`, `_boundary2`,
`:12111-12783`, ~670 lines), the tuner `_tune` (`:12922-13140`),
`_after_attention` / `_pick_expert_backend` (`:13377-13486`), Marlin
fallback (`_marlin_bank`, `_marlin_experts`, `:12783-12856`), and the
`_ma_*` / `_bfp_*` / `_att7` wrappers. `__global__` count (7) is just
the sum of its components' globals.

---

## 2. Duplication catalogue: copy-pasted vs genuinely different

### 2A. Cross-file: decode_layer re-embeds the other three files (~11,300 lines)

| Decode section | Standalone counterpart | `diff` (content-only) | Verdict |
|---|---|---|---|
| `ATT7_CUDA_SRC` 107–4952 (4,846) | attn `CUDA_SRC` 126–3979 (3,854) | 1,790 diff lines; 973 even after `-I` `#`/`//`/blank | **Copy of the pre-refactor version.** Same function set, plus the dead helpers attn_clean removed (`cp_async4z`, `mbar_init`, `mbar_wait`, `tma_load_1d`, `tma_row`, `mma_fp8`, `ex2_poly`, `ex2_e4m3x4_h`, `wgmma_m64n64k16_f16_tb`) and the unfolded `#ifndef OKC_*` wrappers + experiment arms (`OKC_P8EMU`, `OKC_EXH2`, `OKC_PROBE_O8`, …). No new device function in decode's copy (name-set diff: decode ⊃ attn). |
| `MOE_CUDA_SRC` 5117–8619 (3,503) | moe `CUDA_SRC` 189–3643 (3,455) | **162 diff lines** | **Near-identical copy of the pre-refactor version.** Differences are the roadmap comment, the dead helpers moe_clean removed (`dq_hi`, `chunk_mma`, `sk_load_w`, `sk_stage_x`, `wg_pin`, knobs `OKC_FNST/FPF/FCOST2/FCOSTB2/WGSREL32/WGSPROF`), and the `skinny_ring` comment. No new device function. |
| `bfp_*` strings 8890–11162 (2,107) | bf16 strings (2,751 total; TMA 467, GEMM 179, SK2 518, WIDE 361, …) | TMA 11, GEMM 15, GQ 11, MAP 13, MBAR 38, WG 0 diff lines; SK2 376, WIDE 113 | **Copy, with three real divergences** (below). CUDA device/global name sets are identical. |
| Decode Python packers/wrappers | bf16 / moe Python | e.g. `_pack_codes` diff shows decode inlines `_split_nibbles`/`_scatter_code_bits`; `bfp_pack_compressed` inlines `_exponent_window`/`_escapes` | **Copy of pre-refactor Python.** Same logic, minus the refactor's factored helpers. |

The three bfp divergences are the only places decode's copies are *newer*
than the standalone files:

- **SK2 `FUSE` + PDL** (`muse_decode_layer_clean.py:9829-9841, 9900-9974,
  10004, 10157, 10355-10390`): `sk2_kernel<…, PK, FUSE>` takes extra
  `Wsr/outr/part/rcnt/pdl` args; `FUSE` streams 4 extra router stages
  through the same ring; PDL issues the first `P` weight stages before
  `griddepcontrol.wait`. The standalone bf16 `SK2_SRC`
  (`muse_bf16_proj_clean.py:1049-1566`) has neither.
- **WIDE `HLW` removal + spin** (`:10441-10769` vs bf16 `:1570-1928`,
  113 diff lines): decode's `wide_kernel` drops the `HLW` router path
  (SS-wgmma on TMA'd hi/lo tiles, bf16 `:1701-1704`) and the `mapA`
  argument, and uses `okc_spin_ge` + `__threadfence()` where bf16
  uses inline `atom.add.release` / `ld.acquire` loops.
- **`okc_spin_ge`** (`:8962`, used `:9461, 9733, 10692`): a bounded-spin
  helper present in decode's bfp copy, absent from standalone bf16.

*Consequence:* decode cannot be deduped by a mechanical "import the
standalone files" — the copies must first be reconciled (port FUSE/PDL
into bf16's SK2, decide HLW's fate, port `okc_spin_ge`, re-apply the
three refactors' dead-code removals to decode's copies), and the
result still has to live in one file per submission. The ~11,300 lines
are therefore **structural duplication imposed by the one-file rule**,
not author sloppiness — but they are also a maintenance hazard: every
fix must now be made in two places, and the copies have already
drifted in both directions.

### 2B. Within-file families

**B1. wgmma PTX specializations — TEXT duplication, not algorithmic.**
MoE `WG<1..8>` (`muse_moe_w4a16_clean.py:1612-1719`) and `WGSMMA<1..8>`
(`:2490-2604`): each specialization differs only in the `m64n{8·N}k16`
mnemonic and the length of the `%0…` operand list (e.g. `WG<1>` lists
`acc[0][0..3]`, `WG<8>` lists `acc[0..7][0..3]`). Prefill
`wgmma_rs<32,64,128,256,96,160,192,224>`
(`muse_prefill_layer_clean.py:1851-1935`, 85 lines) is the same pattern
with `d[0..N/2-1]`. Attention's four e4m3 wrappers
(`muse_attention_decode_clean.py:572-646`, ~75 lines) are the same
pattern with different A-operand sources (registers vs descriptor,
scale-d on/off, `n72`). *Why the text repeats:* PTX `asm volatile`
operand lists cannot be generated by a C++ loop; the register list
length is part of the asm string. *Why it is mergeable:* bf16 already
does it — `wgmma_rs_defs(bns)` (`muse_bf16_proj_clean.py:312-334`, 23
lines) and `wgmma_ss_defs(bns)` (`:1932-1952`, 21 lines) build the same
strings with `", ".join(f"%{i}" …)`. This is the single clearest
proven abstraction in the corpus (Top-5 #2).

**B2. MoE `skinny_ring` ×2 — near-copy, scheduling-interleaved.**
SKDYN 345 lines vs static 340 lines; `diff` of the two functions is
185 lines, i.e. most of the bodies are shared text. Inside each, the
epilogue is a 3-way `#if OKC_BEPI / OKC_EPI4 / #else` block
(`muse_moe_w4a16_clean.py:1132-1250` and the static counterpart):
three *different* reduction patterns (balanced 8-warp-partial,
4-warp lane-pair, plain) — those three are genuinely different from
each other, but the whole 3-way block appears **twice**, once per ring
variant, with only ~tens of lines of diff. The refactor notes already
flagged this and kept it ("pool-claim logic interleaves with the ring
refill, so a merge would risk changing scheduling") — that is the
correct risk statement; it is also exactly what a `template<bool DYN>`
+ factored `skinny_epi<PB>(…)` would have to preserve (Top-5 #3).

**B3. Prefill `attn_prep` vs `attn_prep2` — half-duplicated.**
`attn_prep_kernel` (`muse_prefill_layer_clean.py:4460-4560`, 101 lines,
`template<int HALVES>`) splits heads across blocks (`blockIdx.y = h`);
`attn_prep2_kernel` (`:4561-4670`, 110 lines) handles both heads in
one block with `vs[2][NK*VP]` smem. `diff` shows the PRMT V-transpose
block (the `__byte_perm` 4×4 transpose, `0x5140/0x7362/0x5410/0x7632`)
is textually identical, and the key-gather loop differs only in
indexing (`c8` vs `c`, `vs[h]` vs `vs`). The divergent part — grid
mapping and smem shape — is real, but it is parameterizable
(Top-5 #4). Note the comment on prep2 (`:4561` ff.) records a measured
result ("batching a thread's 4 page lookups … measured slower:
12.0 → 13.4 µs") — the merge must preserve the *non-batched* issue
order, not just the arithmetic.

**B4. MoE packers — small, mechanical.**
`_pack_codes` vs `_pack_codes_nat`
(`muse_moe_w4a16_clean.py:4090-4106` vs `:4134-4150`): 17 lines each,
differ in 4 lines (the `view` shape, the `permute` order, the
`(_REG_HJ)` vs `((0,0),(1,0),(0,1),(1,1))` table, `word` shape
`[…, :-2]` vs `[…, :-3]`). `_fat_order` vs `_unit_order`
(`:4116-4126` vs `:4126-4134`): genuinely different layouts (fat =
256-row units × 16 k-stages interleaved; unit = per-16-row-block
concatenation) — not mergeable without making the layout itself a
parameter, which is the whole function. `_pack_scales`, `_wgs_copy`
are single-use layouts.

**B5. bf16 12-bit packers — shared core, different layouts.**
`pack_compressed` (`muse_bf16_proj_clean.py:3272-3321`, 50),
`sk2_compress` (`:3495-3533`, 39), `cg_compress` (`:3534-3566`, 33):
all three do exponent-window selection + escape test + 12-bit code
formation; the refactor already factored the first two steps into
`_exponent_window` / `_escapes` (`:3243-3261`). What remains distinct
is the *output layout*: fragment order vs B-fragment tile-units vs
row-major planes. A single packer parameterized by a layout permutation
would save the shared prologue/epilogue (~50 lines) but the permutation
*is* most of each function. Medium confidence, small saving (Top-5 #5).
`build_patch_table` vs `sk2_table` (`:3322-3355` vs `:3621-3644`):
kept separate deliberately in the refactor notes (patch table grouped
by CTA; sk2 table k-ordered within each CTA because escape warps walk
K once) — **genuinely different ordering semantics**, do not merge.
`build_stream` (`:3356-3434`, 79) vs `sk2_stream` (`:3567-3586`, 20):
different stream shapes, not copies.

**B6. Dispatch switches — repeated, but idiomatic and small.**
The 8-case `switch (n)` → `consumer_tile<N>` /
`consumer_item<N>` appears 3× in prefill
(`muse_prefill_layer_clean.py:2265-2282, 2897-2914, 3439-3471`), the
2×8-case `switch (ntc)` → `fat_compute[_wg]<N>` appears in MoE
(`muse_moe_w4a16_clean.py:2002-2020`), and the `attn_fwd` launch switch
in prefill `GLUE` (`:1619-1644`) plus its `cudaFuncSetAttribute` repeat
(`:1664-1688`). C++ cannot dispatch a runtime int to a template
without a switch (or a function-pointer table, which costs an indirect
call in launch code); a fold-expression / macro helper could compress
each switch to ~3 lines, saving ~60–80 lines total across the corpus,
but the switches are *explicit and auditable* and each case list is
the tuning surface (e.g. `MOE_FINE_MASK` arms). Low value; not in Top 5.

**B7. Helpers that look duplicated but are genuinely different.**
- `mbar_*`: attn uses `uint64_t* bar` (pointer) helpers
  (`muse_attention_decode_clean.py:358-386`); bf16/MoE use `uint32_t
  bar` (smem-address) helpers (`muse_bf16_proj_clean.py:215-283`;
  `muse_moe_w4a16_clean.py:830-839`); prefill uses its own inside
  `_CUDA_SRC`. Same names, **incompatible signatures** — merging
  requires changing every call site in every kernel. Do not merge
  across files; within a file each set is already shared.
- `desc_sw128`: the *formula* is identical in attn
  (`muse_attention_decode_clean.py:568-570`) and bf16
  (`make_desc_sw128`, `muse_bf16_proj_clean.py:299-305`), and
  equivalent in MoE (`wg_desc_sw128`, `:1413`) and prefill
  (`desc_sw128`, `:1849` ff.) — a 4-line helper copied 4× under 3
  names. Trivially shareable *if* files shared a header; under the
  one-file rule it is 16 lines of justified duplication.
- `silu` / SwiGLU epilogues: MoE `epi_A`/`epi_B`, WGS epilogue,
  prefill `cit_combine`, bf16 SK2 epilogue all compute
  `silu(gate)*up` — but with different reduction patterns, different
  `normw` handling, different store targets (`inter` vs `ybuf` vs
  `out`). The *formula* is shared; the *epilogue* is not.

**B8. Prefill softmax / pipe helpers — genuinely different, not copies.**
`softmax_core` (`muse_prefill_layer_clean.py:5207-5257`, 51 lines,
simple) vs `softmax_core_rr` (`:5258-5460`, 203 lines, round-robin
turns, named barriers, late-max) vs `softmax_core_p3` (`:5793-5894`,
102) vs `softmax_half` (`:5992-6031`, 40): different pipeline
algorithms, not expansions of one another. Likewise `attn_pipe`
(`:5604`) vs `attn_pipe_eq` (`:5668`) vs `attn_pipe2` (`:5741`) vs
`attn_pipe3` (`:5956`) vs `attn_pipe2h` (`:6093`), and `attn_step`
(`:4887`, 201) vs `attn_step_h` (`:5111`, 96). Prefill's attention
is already factored into these helpers — it is the best-factored
large kernel in the set, and further merging would conflate
algorithms.

**B9. Dead / experiment code still present (small, real).**
- Prefill `boundary2_kernel` (`muse_prefill_layer_clean.py:3990-4094`,
  ~106 lines) + its `CPP` wrapper: dead — not in `_build`'s function
  list (prefill refactor notes confirm). Removable by deletion,
  ~110 lines, no abstraction needed.
- Prefill `GLUE` experiment launch arms (`:1626-1641`, `:1674-1688`):
  `ATTN_NWG1/V52/V53/PIPE_NORR` cases, ~20 lines, guarded by `#ifdef`s
  that are never defined in the file. Same for `ATTN_DBG_NOEXP /
  NOMAX / LB` (`:5265, 5414, 6123`), `MOE_DBG_*` (`:2520, 2544,
  2554, 2560`), attn `OKC_XP_*` runtime checks
  (`muse_attention_decode_clean.py`, 17 `OKC_XP` hits, e.g. `:2098,
  2439, 3103`). The attn refactor notes explicitly left the
  constant-valued `if (OKC_XP_…)` checks unfolded because removing
  them needs a real C++ parser; ptxas already folds them, so they
  cost lines, not perf.
- MoE `BW_SRC` (`muse_moe_w4a16_clean.py:3645-3982`, 338 lines) +
  `_diag_bw` / `_diag` (`:4289-4439`, ~150 lines): diagnostics-only,
  gated by `DIAG=False` / `DIAG_BW=False`. **Not dead** — it is the
  only in-file bandwidth microbenchmark — but it is not part of the
  timed path and could live outside the submission.

---

## 3. Load-bearing vs mergeable — judgement per family

| Family | Load-bearing? | Evidence that would prove a merge safe (GPU ablation) |
|---|---|---|
| MoE 4 `-D` builds (MAIN/TINY/TINY2/MID) | **Yes — already parameterized.** One source, 4 flag tuples. No merge available; the tuples *are* the abstraction. | N/A — but any flag change needs re-timing at M = 4/8/16/17/64 boundaries |
| Attn `Cfg` / `segment` / `MODE` builds | **Yes — already parameterized.** One `segment` template, `Cfg` computes geometry. | N/A |
| bf16 `@@DISPATCH@@` generators | **Yes — already parameterized.** One kernel per family, config tables generate launches. | N/A; config-table edits need per-shape re-timing (the tables cite H200 measurements) |
| Prefill `ATTN_VARIANTS` / `attn_fwd_kernel<…>` | **Yes — already parameterized.** One kernel template, 7 instantiations. | N/A; variant choice is validated by `_validate_attn` (`:770-786`) |
| WG/WGSMMA/wgmma_rs PTX lists (§2B1) | **Text only — mergeable.** The *instruction shapes* (one per N) are load-bearing; the *hand-written text* is not. | Generated source byte-identical to hand-written; SASS identical; per-shape timing unchanged (timing should be a no-op check, not an expectation of speedup) |
| `skinny_ring` SKDYN vs static (§2B2) | **Scheduling is load-bearing; the second copy is not.** Pool-claim interleaving is the risk. | Single-template build with `DYN` as a template param: SASS diff vs both originals + timing at the SKDYN boundary + correctness at chunk boundary M = 384/385 |
| `attn_prep` / `attn_prep2` (§2B3) | **Grid mapping + issue order load-bearing** (measured 12.0→13.4 µs for batched lookups); the transpose + gather code is mergeable. | One-template build: FA3-reference correctness on scored shapes, standalone prep timing (must not regress vs both originals) |
| `_pack_codes` / `_pack_codes_nat` (§2B4) | **Layout is the interface; the 4 differing lines are the layout.** Mergeable mechanically. | CPU-torch: packed words byte-identical on random codes; GPU `prepare()` selftest |
| bf16 12-bit packers (§2B5) | **Layouts load-bearing** (each kernel reads exactly one layout); shared core already factored. | Same CPU-torch byte-identity gate; saving is small, do it only with the packer test |
| `build_patch_table` vs `sk2_table` | **Genuinely different** — CTA-grouped vs k-ordered; merging would change escape-warp walk order. Do not merge. | — |
| Prefill softmax/pipe variants (§2B8) | **Genuinely different algorithms.** Do not merge. | — |
| `mbar_*` / `desc_sw128` helpers (§2B7) | **Conventions differ** (pointer vs smem-address bars); cross-file merge is a signature change, not a dedup. | — |
| Prefill `boundary2`, experiment `#ifdef` arms, attn `OKC_XP_*` runtime checks (§2B9) | **Dead / folded by ptxas.** Deletable, but needs a parser-grade fold for attn, not regexes. | `gcc -E` / SASS identity (attn notes already used a `gcc -E` oracle for the fold) + judged run |
| MoE `BW_SRC` diagnostics | **Safety net, not timed path.** Keep in the repo; it need not be in the submission. | — |

Rule of thumb that emerges: **if two blocks differ only in operand-list
length, a view/permute, or a grid mapping, they are mergeable; if they
differ in *when* something is issued, waited on, or reduced, they are
load-bearing scheduling and merging them is a perf experiment, not a
cleanup.**

---

## 4. Templated design sketch + projection

### 4.1 What the templates would look like

**Wgmma PTX (proven pattern — copy bf16's generator).** Today MoE
hand-writes `WG<1>`…`WG<8>` (108 lines) and `WGSMMA<1..8>` (115
lines); prefill hand-writes `wgmma_rs<…>` (85 lines). Replace all with
the shape bf16 already uses:

```python
# Python generator (cf. muse_bf16_proj_clean.py:312-334, 1932-1952)
def wgmma_defs(ns, mnemonic, acc_fn):
    out = []
    for n in ns:
        nacc = n // 2  # or n*4 for the e4m3 n64 case — the only per-family constant
        regs = ", ".join(f"%{i}" for i in range(nacc))
        cons = ", ".join(f'"+f"(acc[{i}])' for i in range(nacc))
        out.append(f"""
template <> struct WG<{n}> {{  /* or a single template<int N> with enable_if */
  static __device__ __forceinline__ void mma(...) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{8*n}k16.f32.bf16.bf16 "
                 "{{{regs}}}, {{%{nacc}, ...}}, %{nacc+4}, p, 1, 1, 0;");
  }}
}};""")
    return "\n".join(out)
```

One generator (~25 lines) replaces each hand-expanded family. The
per-family differences (accumulator indexing `acc[i][j]` vs `d[i]`,
`mma` vs `mma0` scale-d flag, e4m3 vs bf16 mnemonic) are 2–3
parameters, not 3 separate hand-written families. The generated CUDA
must be byte-identical to today's text — that is checkable on CPU
before any GPU run.

**MoE skinny ring.** One template, pool-claim as policy:

```cpp
template <bool PB, bool DYN>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, …) {
  // … shared prologue / ring refill / main loop …
  if constexpr (DYN) { /* claim next unit from the atomic pool */ }
  else              { /* fixed range v0..v1 */ }
  skinny_epi<PB>(p, S, …);  // the 3-way BEPI/EPI4/#else epilogue, factored once
}
```

The epilogue helper takes the reduction state (`S.red`, `nwc0/nwc1`,
`rb`, `j`, `i`, `two`) explicitly — its signature will be long, which
is the honest cost of the factoring; the alternative (leaving 2 copies)
is also defensible, and the notes already chose it once.

**Prefill prep.** One template over the existing axis:

```cpp
template <int HALVES, bool SPLIT_HEADS>  // SPLIT_HEADS=false is today's prep2
__global__ void attn_prep_kernel(…);
```

`SPLIT_HEADS` selects `blockIdx.y = h` + `vs[NK*VPS]` vs
`vs[2][NK*VP]`; the PRMT transpose and gather loops are shared.

**MoE packers.**

```python
def _pack_codes(codes, rows, natural=False):
    view, perm, table, tail = ((nb,2,8,8,8,4,4,4), (0,3,4,2,5,6,1,7), _REG_HJ, -2) if not natural else \
                               ((nb,2,8,8,8,4,2,4,2), (0,3,4,2,7,5,1,6,8), ((0,0),(1,0),(0,1),(1,1)), -3)
```

**bf16 dispatch (already the template — keep).** The `@@DISPATCH@@`
+ config-table pattern in bf16 (`_tma_srcs`, `_gemm_src`, …) *is* the
templated design the other files approximate with C++ templates
(attn `Cfg`, prefill `attn_fwd_kernel<…>`, MoE `-D` builds). No new
abstraction is needed there; the lesson is that the corpus already
contains its own answer, twice over (C++ templates for in-kernel
variation, Python generators for launch-table variation).

### 4.2 Projection (speculation, labelled as such)

| Change | Today | After | Saved |
|---|---|---|---|
| Decode: generate from standalone files (rule change) | 13,486 | ~2,200 (glue + Triton + tuner only) | ~11,300 |
| Wgmma PTX → generators (MoE + prefill + attn) | ~308 | ~55 | ~250–290 |
| `skinny_ring` → one template + factored epilogue | 685 | ~440 | ~200–250 |
| `attn_prep` + `attn_prep2` → one template | 211 | ~160 | ~50 |
| Packers → layout-parameterized (`_pack_codes`, bf16 trio) | ~160 | ~95 | ~65 |
| Delete `boundary2` + experiment `#ifdef` arms (prefill) | ~130 | 0 | ~130 |
| **Total if everything landed** | **33,328** | **~21,500** | **~11,800** |
| **Total without the decode rule change** | **33,328** | **~32,600** | **~700** |

The last row is the important one: **inside the current rules, the
provable abstraction saving is ~700 lines (~2%)**. The rest of the
" saving" is not an abstraction at all — it is deleting the forced
copy in decode, which requires either a shared-module rule or a
generation step that assembles `decode_layer` from the three
standalone files at build time (the refactor notes' builder script
`scratch-script` is already a precedent for generated
clean files, but it generated a *cleaning*, not an *assembly*).

### 4.3 What the templated design does *not* change

Compiled code should be identical: these are text-level templates
over the same instantiations. Any SASS difference after a merge is a
bug in the merge, not an improvement. None of this is a speedup
project; if a merge changes timing, the merge is wrong.

---

## 5. Genuinely irreducible (do not abstract away)

1. **`segment()`** (`muse_attention_decode_clean.py:1282-~3215`, ~1,930
   lines) — one per-segment main loop whose branches share dozens of
   locals (pipeline stages, mbarrier phases, producer/consumer
   hand-off). The refactor notes already declined to split it; splitting
   it is a behaviour-risk change, not a cleanup.
2. **MoE `skinny_ring` / `fat_stream` / WGS main loops** — the
   scheduling *is* the kernel: ring refill, pool claims, helper-warp
   issue, epilogue interleaving. The code is long because the schedule
   is long.
3. **Prefill `attn_fwd_kernel` + `attn_step` / `softmax_core_rr` /
   `attn_pipe*`** — distinct pipeline algorithms with different
   barrier/turn structure; they share helpers already.
4. **bf16 `sk2` / `wide` / `skinny_tma` / `gemm90` kernels** — distinct
   data paths (TMA ring vs cp.async, 12-bit compressed vs BF16, escape
   correction vs none). Same for `rgemv`/`cgemv`/`sgemv`: three
   input formats, three kernels.
5. **Config tables and `-D` tuples** — `TMA_CONFIGS`,
   `GEMM_CONFIGS`, `SK2_CONFIGS`, `WIDE_CONFIGS`, `GQ_CONFIGS`,
   MoE `MAIN/TINY/TINY2/MID_DEFINES`, attn `OKC_*` table, prefill
   `ATTN_VARIANTS`: these are *measured values*, not code. Each entry
   is a distinct tuned operating point; "abstracting" them away would
   mean deleting tuning data. The per-file comments citing H200
   measurements are the audit trail for these tables.
6. **Packing layouts** — each packer produces a byte layout a specific
   kernel reads. The layout is the kernel's input contract; you can
   share the prologue, not the layout.
7. **PTX instruction shapes** — one `wgmma.m64n{N}k16` per N must exist
   in the binary. Generation removes *text*, not *shapes*.
8. **Diagnostics / self-test / autotune** — MoE `BW_SRC` (338 lines)
   + `_diag`, bf16 `_selftest`/`_autotune` (`:3890-4190`), prefill
   `_selftest*` (`:1068-1250`), decode `_tune` (`:12922-13140`).
   These are the only in-file evidence a kernel is correct on a GPU
   the author did not test on. They are not on the timed path.
   (Prefill `boundary2` is the exception — it is genuinely dead,
   §2B9.)
9. **The one-file duplication itself** — while each submission must be
   self-contained, decode's copies are irreducible *within the rules*.
   The fix is a rule/generation change, not a refactor.
10. **WHY comments on tuned choices** — e.g. the `OKC_*` table
    comments, the `MOE_FINE_MASK` / `ATTN_LATEMAX` notes, the
    "measured slower" parentheticals. They look like history; they
    are the reason each default is what it is. The refactors already
    removed the pure measurement logs (dates, timings); what remains
    is justification.

---

## Bottom line

- **Are we sure everything is needed?** For the *unique* code: mostly
  yes. ~22k unique lines, of which ~19k are CUDA code, and the bulk of
  that is either one templated kernel per family, its instantiations,
  or a distinct data path / schedule. The "5k lines" intuition
  undercounts by 6× and, more importantly, mistakes *instantiation*
  for *duplication*.
- **Is there abstraction needed?** A little, in specific places:
  generate the wgmma PTX lists (the pattern already exists in bf16),
  unify `skinny_ring`, `attn_prep`, and the smallest packers, delete
  `boundary2` + dead experiment arms. ~700 lines, high-to-medium
  confidence, each gateable by byte-identical generation + SASS diff +
  per-shape timing.
- **The big one is structural, not textual:** decode_layer duplicates
  the other three submissions because the format forces it. If that
  is the duplication that bothers you, the fix is to let a submission
  be *generated* from shared sources (or to accept the copy and keep
  it in sync — which is a process decision, with the divergences in
  §2A as the current cost of not having one).
- **What not to do:** do not merge kernels that differ in scheduling
  (`skinny_ring` pool-claim, prefill softmax/pipe variants, MoE
  epilogue reduction patterns, `build_patch_table` vs `sk2_table`),
  do not delete selftests/diagnostics to save lines, and do not treat
  config tables as duplication.
