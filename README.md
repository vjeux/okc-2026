# Open Kernel Challenge 2026 — Muse Spark kernel entries (refactored)

Five single-file CUDA/Python kernels for the [Open Kernel Challenge](https://openkernelchallenge.dev) (Muse Spark), 2026 kernel track. Apache 2.0.

This is the refactored, GPU-validated submission set: the same kernels as the
original entries, cleaned up and documented, with behavior-identical
refactors. Each file is a complete OKC submission: one Python file, native
CUDA via `load_inline`, CUDA-graph capturable. The judge runs them against
SGLang baselines on H200s. Per-file refactor notes
(`muse_*_REFACTOR_NOTES.md`) describe what changed and what was verified;
`ABSTRACTION_ANALYSIS.md` analyzes duplication and abstraction across the set.

| Problem | File | Clean score | Board status | Scored job | Judge-check job |
|---|---|---|---|---|---|
| `muse_decode_layer` | `muse_decode_layer.py` | **Scored 2.0939** | Current board entry, in clean code | `ad1a2885dcc44ecb` | `c6582ca5702c4d12` |
| `muse_moe_w4a16` | `muse_moe_w4a16.py` | **Scored 2.3267** | Current board entry, in clean code | `06beba86b8784027` | `996937572b454a76` |
| `muse_attention_decode` | `muse_attention_decode.py` | Clean scored 2.5806 | Board stays v298b (2.5976) | `6a7c9e676fbc43d0` | `e78062041c924361` |
| `muse_prefill_layer` | `muse_prefill_layer.py` | Clean scored 1.7778 | Board stays 1.7982 | `6379f7bafcc9493c` | `7dc20987b20b4754` |
| `muse_bf16_proj` | `muse_bf16_proj.py` | Clean scored 1.4905 | Board stays 1.5014 | `48d90a0960b14304` | `6d606bc237bc4f51` |

Notes:
- Decode and MoE are current board entries in clean code: the scored runs
  above were the refactored files themselves.
- Attention, prefill, and bf16 are validated equivalents: their clean
  versions scored within ~0.7% of the board entries, and the board entries
  stay in place.
- All five clean files passed the judge checks (check job IDs in the table
  above).
