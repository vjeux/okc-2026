# Open Kernel Challenge 2026 — Muse Spark kernel entries (team vjeux)

Five single-file CUDA/Python kernels for the [Open Kernel Challenge](https://openkernelchallenge.dev) (Muse Spark), 2026 kernel track. Apache 2.0.

Each file is a complete OKC submission: one Python file, native CUDA via `load_inline`, CUDA-graph capturable. The judge runs them against SGLang baselines on H200s.

| Problem | File | Judge score | Status |
|---|---|---|---|
| `muse_attention_decode` | `muse_attention_decode.py` (v298) | Check 2.5810 (+0.88% vs v238's check) | **Best file, not on the board** — its scored run was withdrawn before it ran, and the judge deduplicates by file hash, so it cannot be resubmitted. Board entry remains v238 (2.5896). |
| `muse_decode_layer` | `muse_decode_layer.py` (v157e) | **Scored 2.0860** (job `51521bb48c3b4fa3`), +1.55% over v154 (2.0541) | Board entry |
| `muse_moe_w4a16` | `muse_moe_w4a16.py` (v216x) | **Scored 2.3245** (job `13b0d9a4a61f4035`) | Board entry |
| `muse_prefill_layer` | `muse_prefill_layer.py` (v116s) | **Scored 1.798** (job `c0a1e9ad649345c5`) | Board entry |
| `muse_bf16_proj` | `muse_bf16_proj.py` (v141) | **Scored 1.5014** (job `08c28ade42034919`) | Board entry |

Notes:
- Final board (Oct 27, 2026): attention #14, prefill #16, moe #31, decode #45, bf16 #122. No category won.
- The v298 / v157e / v265x judge checks, the v158b reversal (a pod-local +0.91% that lost −11% on the judge), and the routing-sensitivity lesson are documented in the commit history of this repo's development, not in this snapshot.
- All five files were scanned for internal strings before publication; they contain only the kernel source.
