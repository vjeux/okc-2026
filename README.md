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
