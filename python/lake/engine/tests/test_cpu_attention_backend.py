"""Cpu attention backend tests."""

from __future__ import annotations

import torch

from lake.engine.model_executor.layers.attentions import (
    AttentionMetadata,
    CpuAttentionBackend,
)
from lake.engine.model_executor.models.qwen.qwen3 import Qwen3PagedAttention


def _manual_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    mask = torch.ones(q_len, k_len, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def test_cpu_attention_backend_matches_manual_causal_gqa() -> None:
    q = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]]],
        dtype=torch.float32,
    )
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]], dtype=torch.float32)

    out = CpuAttentionBackend().forward_tensors(q, k, v)
    expected = _manual_causal_attention(q, k, v)
    torch.testing.assert_close(out, expected)


def test_qwen3_paged_attention_uses_cpu_backend_for_tensors() -> None:
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)
    attn = Qwen3PagedAttention(
        num_heads=2, head_dim=4, num_kv_heads=1, backend=CpuAttentionBackend()
    )

    out = attn(q, k, v)
    expected = CpuAttentionBackend().forward_tensors(q, k, v, scale=4**-0.5)
    torch.testing.assert_close(out, expected)


def test_qwen3_paged_attention_dispatches_varlen() -> None:
    """``Qwen3PagedAttention(q,k_cache,v_cache,attn_meta=...)`` 走 forward_varlen 分派。

    混合批：req0 extend（prefill，q_len=seq_len=5）；req1 decode（q_len=1，seq_len=4）。
    对照 ``_manual_padded_causal`` 参照，验证经模型层分派后仍与 CPU 后端直连等价。
    """
    torch.manual_seed(4)
    H, Hkv, D = 2, 1, 6
    scale = D**-0.5
    seq_lens = [5, 4]
    q_lens = [5, 1]
    q_per_req = [torch.randn(ql, H, D) for ql in q_lens]
    k_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    v_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    k_cache, tables = _build_paged_kv(k_per_req, block_size=8)
    v_cache, _ = _build_paged_kv(v_per_req, block_size=8)
    q = torch.cat(q_per_req, dim=0)
    meta = _meta(seq_lens, q_lens, tables)

    attn = Qwen3PagedAttention(
        num_heads=H, head_dim=D, num_kv_heads=Hkv, backend=CpuAttentionBackend(), block_size=8
    )
    out = attn(q, k_cache, v_cache, attn_meta=meta)

    expected = torch.cat(
        [_manual_padded_causal(q_per_req[i], k_per_req[i], v_per_req[i], seq_lens[i], scale=scale)
         for i in range(len(seq_lens))],
        dim=0,
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    direct = CpuAttentionBackend().forward_varlen(q, k_cache, v_cache, meta, block_size=8, scale=scale)
    torch.testing.assert_close(out, direct, rtol=1e-6, atol=1e-6)


def _manual_varlen_causal(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """q [q_len, H, D], k/v [kv_len, Hkv, D] → out [q_len, H, D]，因果（q 全段对全 KV）。"""
    if q.shape[1] != k.shape[1]:
        repeat = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    # head 维移到最前，做 [H, q_len, D] @ [H, D, kv_len]
    qh = q.movedim(1, 0)  # [H, q_len, D]
    kh = k.movedim(1, 0)  # [H, kv_len, D]
    vh = v.movedim(1, 0)  # [H, kv_len, D]
    sc = scale if scale is not None else q.shape[-1] ** -0.5
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * sc  # [H, q_len, kv_len]
    q_len = q.shape[0]
    k_len = k.shape[0]
    mask = torch.ones(q_len, k_len, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)  # [H, q_len, D]
    return out.movedim(0, 1)  # [q_len, H, D]


def _manual_padded_causal(
    q_i: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: int, scale: float | None = None
) -> torch.Tensor:
    """统一参照：把 query pad 进 [prefix_len:prefix_len+q_len] 再因果，slice 出新段。

    与 ``forward_varlen`` 的 extend 路径等价；decode（q_len=1, prefix=seq_len-1）
    经此参照化为「最后位置因果」= full-visible，与 decode 路径（is_causal=False）一致。
    """
    q_len = q_i.shape[0]
    prefix_len = seq_len - q_len
    H, D = q_i.shape[1], q_i.shape[2]
    padded = torch.zeros(seq_len, H, D, dtype=q_i.dtype)
    padded[prefix_len:] = q_i
    full = _manual_varlen_causal(padded, k, v, scale=scale)
    return full[prefix_len:prefix_len + q_len]


def _build_paged_kv(
    kv_per_req: list[torch.Tensor], block_size: int
) -> tuple[torch.Tensor, list[list[int]]]:
    """把每请求的 KV [seq_len, Hkv, D] 放进 flat slot 池，返回 (cache, block_tables)。

    每请求独占连续 block，block 索引按请求顺序递增；slot = block*block_size + offset，
    与 ``forward_varlen`` 的 gather 逻辑严格一致。
    """
    num_reqs = len(kv_per_req)
    total_blocks = sum((t.shape[0] + block_size - 1) // block_size for t in kv_per_req)
    total_slots = total_blocks * block_size
    hkv = kv_per_req[0].shape[1]
    d = kv_per_req[0].shape[2]
    cache = torch.zeros(total_slots, hkv, d, dtype=kv_per_req[0].dtype)
    tables: list[list[int]] = []
    blk = 0
    for kv in kv_per_req:
        seq_len = kv.shape[0]
        nblocks = (seq_len + block_size - 1) // block_size
        for t in range(seq_len):
            slot = (blk + t // block_size) * block_size + (t % block_size)
            cache[slot] = kv[t]
        tables.append(list(range(blk, blk + nblocks)))
        blk += nblocks
    return cache, tables


def _meta(
    seq_lens: list[int],
    q_lens: list[int],
    tables: list[list[int]],
) -> AttentionMetadata:
    qsl = [0]
    for ql in q_lens:
        qsl.append(qsl[-1] + ql)
    return AttentionMetadata(
        seq_lens={str(i): s for i, s in enumerate(seq_lens)},
        query_start={str(i): seq_lens[i] - q_lens[i] for i in range(len(seq_lens))},
        query_end={str(i): seq_lens[i] for i in range(len(seq_lens))},
        block_tables={str(i): tables[i] for i in range(len(tables))},
        block_table_tensor=tables,
        slot_mapping=[],
        positions=[],
        query_start_loc=qsl,
        seq_lens_ordered=list(seq_lens),
        max_seq_len=max(seq_lens) if seq_lens else 0,
        max_query_len=max(q_lens) if q_lens else 0,
        num_reqs=len(seq_lens),
        num_actual_tokens=qsl[-1] if qsl else 0,
    )


def test_forward_varlen_extend_matches_causal() -> None:
    torch.manual_seed(0)
    H, Hkv, D = 4, 2, 8
    seq_lens = [5, 3]
    scale = D**-0.5
    q_per_req = [torch.randn(s, H, D) for s in seq_lens]
    k_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    v_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    k_cache, tables = _build_paged_kv(k_per_req, block_size=8)
    v_cache, _ = _build_paged_kv(v_per_req, block_size=8)
    q = torch.cat(q_per_req, dim=0)
    meta = _meta(seq_lens, seq_lens, tables)

    out = CpuAttentionBackend().forward_varlen(q, k_cache, v_cache, meta, scale=scale)

    expected = torch.cat(
        [_manual_varlen_causal(q_per_req[i], k_per_req[i], v_per_req[i], scale=scale)
         for i in range(len(seq_lens))],
        dim=0,
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_forward_varlen_decode_matches_full_visible() -> None:
    torch.manual_seed(1)
    H, Hkv, D = 4, 2, 8
    seq_lens = [6, 4]
    scale = D**-0.5
    # decode：q_len=1，prefix=seq_len-1
    q_per_req = [torch.randn(1, H, D) for _ in seq_lens]
    k_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    v_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    k_cache, tables = _build_paged_kv(k_per_req, block_size=8)
    v_cache, _ = _build_paged_kv(v_per_req, block_size=8)
    q = torch.cat(q_per_req, dim=0)
    meta = _meta(seq_lens, [1, 1], tables)

    out = CpuAttentionBackend().forward_varlen(q, k_cache, v_cache, meta, scale=scale)

    expected = torch.cat(
        [_manual_padded_causal(q_per_req[i], k_per_req[i], v_per_req[i], seq_lens[i], scale=scale)
         for i in range(len(seq_lens))],
        dim=0,
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_forward_varlen_mixed_batch() -> None:
    torch.manual_seed(2)
    H, Hkv, D = 2, 1, 6
    scale = D**-0.5
    # req0 extend（prefill，q_len=seq_len=5）；req1 decode（q_len=1，seq_len=4）
    seq_lens = [5, 4]
    q_lens = [5, 1]
    q_per_req = [torch.randn(ql, H, D) for ql in q_lens]
    k_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    v_per_req = [torch.randn(s, Hkv, D) for s in seq_lens]
    k_cache, tables = _build_paged_kv(k_per_req, block_size=8)
    v_cache, _ = _build_paged_kv(v_per_req, block_size=8)
    q = torch.cat(q_per_req, dim=0)
    meta = _meta(seq_lens, q_lens, tables)

    out = CpuAttentionBackend().forward_varlen(q, k_cache, v_cache, meta, scale=scale)

    expected = torch.cat(
        [_manual_padded_causal(q_per_req[i], k_per_req[i], v_per_req[i], seq_lens[i], scale=scale)
         for i in range(len(seq_lens))],
        dim=0,
    )
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_forward_varlen_chunked_extend() -> None:
    """extend 但 q_len < seq_len（chunked prefill：已有 prefix KV，本步算新段）。"""
    torch.manual_seed(3)
    H, Hkv, D = 2, 2, 6
    scale = D**-0.5
    seq_lens = [7]
    q_lens = [3]  # prefix_len=4，本步算 token[4:7)
    q_per_req = [torch.randn(3, H, D)]
    k_per_req = [torch.randn(7, Hkv, D)]
    v_per_req = [torch.randn(7, Hkv, D)]
    k_cache, tables = _build_paged_kv(k_per_req, block_size=8)
    v_cache, _ = _build_paged_kv(v_per_req, block_size=8)
    q = q_per_req[0]
    meta = _meta(seq_lens, q_lens, tables)

    out = CpuAttentionBackend().forward_varlen(q, k_cache, v_cache, meta, scale=scale)

    # 参照：新 query 段对全 KV 因果（query 位置 = 4,5,6）
    full_q = torch.zeros(7, H, D, dtype=q.dtype)
    full_q[4:] = q
    expected = _manual_varlen_causal(full_q, k_per_req[0], v_per_req[0], scale=scale)[4:]
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
