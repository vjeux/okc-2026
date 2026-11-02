# Open Kernel Challenge 2026

Five OKC kernel submissions for Muse Spark, one Python file per problem. Each file is a complete, self-contained submission.

| File | Problem |
|---|---|
| `muse_attention_decode.py` | Gated attention decode: a fused kernel computing `rmsnorm(softmax(Q K^T * scale) V) * sigmoid(gate)`, with Q/K RMS-norm, RoPE, and an FP8 KV cache. |
| `muse_bf16_proj.py` | Dense projections: `out = x @ W^T` for the five projection matrices (`qkv_gate`, `o`, `pooler`, `unpooler`, `router`), with losslessly compressed BF16 weights. |
| `muse_decode_layer.py` | Decode layer: one decode step — QKV projection, attention, O projection and residual boundaries, pooler and router, routed experts, and unpooler. |
| `muse_moe_w4a16.py` | Routed experts: NVFP4 W4A16 mixture-of-experts, a gated MLP per active expert with top-8 routing over 256 experts. |
| `muse_prefill_layer.py` | Prefill layer: one transformer layer step over a batch of tokens — QKV projection, Q/K preparation, attention, O projection and boundary, router and pooler, grouped MoE, and combine, unpooler, and second boundary. |

## Building and using

Each file is native CUDA, compiled with `torch.utils.cpp_extension.load_inline`. Files that carry weights expose `prepare(weights)` to build state once; every file exposes `kernel_fn`, which runs one step (`kernel_fn(data)` for attention decode, `kernel_fn(data, state)` for the others).

## License

Apache 2.0 — see `LICENSE`.

## Attention regression tests

Install PyTorch and pytest, then run `python -m pytest -q tests/test_attention_decode.py`.
CPU tests exercise the PyTorch fallback against an independent, per-query reference
and an analytic uniform-attention case. Native tests require an sm_90a GPU and a
CUDA toolkit; they are skipped on other hardware. They cover page-end appends,
scattered pages, causal window edges, new-cache writes, empty trailing splits,
and repeated launches through default, plain split and cooperative plans.
A separate CUDA test checks the actual BF16 rounding helper at rounding ties.

The native numerical checks allow FP8 query error: output relative L2 is bounded
by `clamp(2.5 * fp8_query_error + 0.015, 0.03, 0.15)`, with a per-row guard;
new cache entries allow relative L2 of 0.08. These tests do not require bitwise
agreement with the FP32 reference or establish a performance improvement.
CPU results alone do not validate CUDA compilation, synchronization, CUDA graph
replay, or native accuracy. Cluster/flat plans and full benchmark shapes still
need GPU validation beyond this small regression suite.
