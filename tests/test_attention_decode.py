"""CPU fallback checks and opt-in-by-hardware sm_90a kernel regressions.

The scalar reference slices each query's logical interval, independently of the
kernel's page-tile/split traversal. CPU success does not validate native CUDA.
"""
import re

import pytest
import torch

import muse_attention_decode as attention


FP8 = torch.float8_e4m3fn
HOPPER = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


def make_data(seq=63, length=8, page=64, window=0, scale=1.0, uniform=False):
    gen = torch.Generator().manual_seed(813)
    hq, hkv, d = 16, 2, 64
    pages = (seq + length + page - 1) // page
    # An extra physical page and reversed mapping expose accidental contiguous
    # addressing. The unused logical table entry has the scheduler's -1 sentinel.
    table = torch.tensor([list(range(pages, 0, -1)) + [-1]], dtype=torch.int32)
    positions = torch.arange(seq, seq + length)
    slots = table[0, positions // page].long() * page + positions % page
    qkv = torch.randn(1, length, (2 * hq + 2 * hkv) * d, generator=gen).bfloat16()
    if uniform:
        qkv[..., :hq * d] = 0
        qkv[..., (hq + 2 * hkv) * d:] = 0
    shape = (pages + 1, page, hkv, d)
    k = torch.randn(shape, generator=gen).to(FP8)
    v = torch.randn(shape, generator=gen).to(FP8)
    return dict(
        qkv=qkv, positions=positions, cache_loc=slots,
        cache_seqlens=torch.tensor([seq + length], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32),
        scheduler_metadata=None, k_cache=k, v_cache=v,
        k_scale=torch.tensor(scale), v_scale=torch.tensor(scale),
        page_table=table, seq_lens=torch.tensor([seq], dtype=torch.int32),
        config=dict(batch=1, q_len=length, num_q_heads=hq, num_kv_heads=hkv,
                    head_dim=d, page_size=page, window=window,
                    rope_theta=500000.0 if window else 0.0,
                    qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=scale),
    )


def clone_data(data, device="cpu"):
    return {k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()}


def rms(x):
    x = x.float()
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)


def prepared(data, fp8_query=False):
    c = data["config"]
    length = c["q_len"]
    q, k, v, gate = data["qkv"].float().split([1024, 128, 128, 1024], -1)
    q, k = rms(q.view(length, 16, 64)), rms(k.view(length, 2, 64))
    if not fp8_query:
        q = q * c["qk_scale_factor"]
    if c["rope_theta"]:
        angles = data["positions"].float()[:, None] / c["rope_theta"] ** (torch.arange(32) / 32)
        rotation = torch.polar(torch.ones_like(angles), angles)[:, None, :]
        q = torch.view_as_real(torch.view_as_complex(q.reshape(length, 16, 32, 2)) * rotation).flatten(-2)
        k = torch.view_as_real(torch.view_as_complex(k.reshape(length, 2, 32, 2)) * rotation).flatten(-2)
    q = q.to(FP8 if fp8_query else torch.bfloat16).float()
    if fp8_query:
        q *= c["qk_scale_factor"]
    return q, k, v.view(length, 2, 64), gate.view(length, 16, 64)


def reference(data, k_cache, v_cache, fp8_query=False, round_output=True):
    """Attend the implementation's own new KV, as the OKC accuracy check does."""
    c = data["config"]
    q, _, _, gate = prepared(data, fp8_query)
    out = []
    for j, pos in enumerate(data["positions"].tolist()):
        lo = max(0, pos - c["window"] + 1) if c["window"] else 0
        logical = torch.arange(lo, pos + 1)
        rows = data["page_table"][0, logical // c["page_size"]].long() * c["page_size"] + logical % c["page_size"]
        keys = (k_cache.view(-1, 2, 64)[rows].float() * data["k_scale"]).repeat_interleave(8, dim=1)
        vals = (v_cache.view(-1, 2, 64)[rows].float() * data["v_scale"]).repeat_interleave(8, dim=1)
        scores = (keys * q[j]).sum(-1).T * c["sm_scale"]
        a = (scores.softmax(-1).T[..., None] * vals).sum(0)
        out.append((rms(a.bfloat16() if round_output else a) * gate[j].sigmoid()).bfloat16())
    return torch.stack(out).reshape(1, c["q_len"], 1024)


def relative_l2(got, want):
    return float(torch.linalg.vector_norm(got.float() - want.float()) /
                 torch.linalg.vector_norm(want.float()).clamp_min(1e-12))


def check_cache(before, after):
    _, k, v, _ = prepared(before)
    slots = before["cache_loc"]
    for name, values, scale in (("k_cache", k, before["k_scale"]),
                                ("v_cache", v, before["v_scale"])):
        old, new = before[name].view(-1, 2, 64), after[name].cpu().view(-1, 2, 64)
        expected = (values / scale).to(FP8).float()
        assert relative_l2(new[slots], expected) <= 0.08
        untouched = torch.ones(old.shape[0], dtype=torch.bool)
        untouched[slots] = False
        torch.testing.assert_close(new.view(torch.uint8)[untouched], old.view(torch.uint8)[untouched], rtol=0, atol=0)


# Page end before/at/after append, page-size-1 scatter, causal window edges,
# and a short context with more requested splits than nonempty 64-key tiles.
CASES = [
    pytest.param(63, 8, 64, 0, id="page-end-crossing"),
    pytest.param(64, 1, 64, 0, id="page-start"),
    pytest.param(65, 8, 64, 0, id="page-start-plus-one"),
    pytest.param(63, 8, 1, 0, id="scattered-rows"),
    pytest.param(2050, 8, 64, 2048, id="window-edges"),
    pytest.param(10, 8, 64, 2048, id="empty-splits"),
]


@pytest.mark.parametrize("seq,length,page,window", CASES)
@pytest.mark.parametrize("scale", [1.0, 1 / 64])
def test_fallback_boundaries(seq, length, page, window, scale):
    data = make_data(seq, length, page, window, scale)
    actual_data = clone_data(data)
    actual = attention._fallback(actual_data)
    expected = reference(data, actual_data["k_cache"], actual_data["v_cache"])
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.004)
    check_cache(data, actual_data)


@pytest.mark.skipif(not HOPPER, reason="requires sm_90a CUDA GPU and CUDA toolkit")
@pytest.mark.parametrize("seq,length,page,window", CASES)
@pytest.mark.parametrize("scale", [1.0, 1 / 64])
@pytest.mark.parametrize("mode", ["default", "split", "cooperative"])
def test_native_boundaries(seq, length, page, window, scale, mode):
    data = make_data(seq, length, page, window, scale)
    overrides = {} if mode == "default" else dict(flat=0, cs=0, n_splits=8, nc=1, coop=int(mode == "cooperative"))
    # Two calls also exercise workspace reset; each call receives fresh tensors.
    for _ in range(2):
        gpu = clone_data(data, "cuda")
        actual = attention._launch(gpu, **overrides).cpu()
        check_cache(data, gpu)
        k, v = gpu["k_cache"].cpu(), gpu["v_cache"].cpu()
        expected = reference(data, k, v, round_output=False)
        fp8 = reference(data, k, v, fp8_query=True, round_output=False)
        # OKC's query-dependent relative-L2 budget, not bitwise BF16 equality.
        tolerance = min(0.15, max(0.03, 2.5 * relative_l2(fp8, expected) + 0.015))
        assert torch.isfinite(actual).all()
        assert relative_l2(actual, expected) <= tolerance
        errors = torch.linalg.vector_norm((actual.float() - expected.float()).reshape(-1, 64), dim=-1)
        norms = torch.linalg.vector_norm(expected.float().reshape(-1, 64), dim=-1)
        assert torch.all(errors <= 5 * tolerance * norms.clamp_min(norms.mean() / 10))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not HOPPER, reason="requires sm_90a CUDA GPU and CUDA toolkit"))])
@pytest.mark.parametrize("window", [0, 2048])
def test_uniform_attention_includes_exact_interval(window, device):
    data = make_data(seq=2050, window=window, uniform=True)
    # A distinctive value at each window edge and each new token makes a
    # skipped oldest key, included expired key, or future token observable.
    values = torch.zeros_like(data["v_cache"], dtype=torch.float32).view(-1, 2, 64)
    for pos, value in [(0, 32), (2, -128), (3, 128)]:
        row = int(data["page_table"][0, pos // 64]) * 64 + pos % 64
        values[row, :, 0] = value
    values[..., 1] = 1
    data["v_cache"] = values.reshape(data["v_cache"].shape).to(FP8)
    new_v = data["qkv"][..., 1152:1280].view(8, 2, 64)
    new_v.zero_()
    new_v[..., 1] = 1
    new_v[..., 2] = torch.arange(1, 9)[:, None]
    run = attention._fallback if device == "cpu" else attention.kernel_fn
    actual = run(clone_data(data, device)).cpu().view(8, 16, 64)
    for j in range(8):
        # With q == 0 and gate == 0, the result is RMS(mean(values)) / 2.
        count = 2048 if window else 2051 + j
        mean = torch.zeros(64)
        mean[0] = (128 if window and j == 0 else 0) / count if window else 32 / count
        mean[1] = 1
        mean[2] = ((j + 1) * (j + 2) / 2) / count
        expected = (rms(mean.bfloat16()) / 2).bfloat16()
        torch.testing.assert_close(actual[j], expected.expand(16, -1), rtol=0.01, atol=0.001)


@pytest.mark.skipif(not HOPPER, reason="requires sm_90a CUDA GPU and CUDA toolkit")
def test_native_bf16_rounding_helper():
    # Execute the actual device helper, not a Python approximation of it. This
    # isolates the BF16 boundary from attention's deliberately tolerant checks.
    from torch.utils.cpp_extension import load_inline

    helper = re.search(
        r"__device__ __forceinline__ float round_bf16\(float x\) \{[^}]+\}",
        attention.CUDA_SRC,
    ).group()
    module = load_inline(
        name="okc_attention_bf16_boundary_test",
        cpp_sources="torch::Tensor rounding(torch::Tensor x);",
        cuda_sources="""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
""" + helper + """
__global__ void apply(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = round_bf16(x[i]);
}
torch::Tensor rounding(torch::Tensor x) {
  auto y = torch::empty_like(x);
  apply<<<(x.numel() + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
  return y;
}
""",
        functions=["rounding"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a"],
    )
    # Adjacent BF16 midpoints, both tie directions and signs, and activation
    # magnitudes before/after a 1/64 cache scale.
    x = torch.cat([torch.linspace(-8, 8, 10001),
                   torch.tensor([1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, -1 - 3 / 256])]).cuda()
    for scale in (1.0, 64.0):
        values = x * scale
        torch.testing.assert_close(module.rounding(values), values.bfloat16().float(), rtol=0, atol=0)
