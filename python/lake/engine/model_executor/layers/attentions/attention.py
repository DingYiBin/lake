"""Attention 后端选择（对照 vLLM AttentionBackend；agent 出表、runner 出其余 metadata 留 D4）。

后端矩阵：
- ``CpuAttentionBackend``：纯 torch SDPA，CPU/dev/test 验证路径（方案 A，对齐 SGLang ``torch_native``）。
- ``FlashAttn2Backend``：上游 ``flash-attn`` 包 FA2 paged varlen，GPU 生产路径（非入图）。
- ``RefAttentionBackend`` / ``TritonAttentionBackend``：早期 list-based 原型，保留兼容。
"""

from __future__ import annotations

from typing import List, Protocol

import torch

from lake.kernels.attn_ref import causal_attn, causal_attn_queries
from lake.kernels.attn_triton import causal_attn_triton
from lake.engine.model_executor.layers.attentions.attention_metadata import (
    AttentionMetadata,
)


class AttentionBackend(Protocol):
    name: str

    def forward(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
    ) -> List[List[float]]: ...

    def forward_queries(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
        q_pos_start: int,
    ) -> List[List[float]]: ...


class RefAttentionBackend:
    name = "ref"

    def forward(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
    ) -> List[List[float]]:
        return causal_attn(q, k, v)

    def forward_queries(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
        q_pos_start: int,
    ) -> List[List[float]]:
        return causal_attn_queries(q, k, v, q_pos_start)


class TritonAttentionBackend:
    name = "triton"

    def forward(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
    ) -> List[List[float]]:
        return causal_attn_triton(q, k, v)

    def forward_queries(
        self,
        q: List[List[float]],
        k: List[List[float]],
        v: List[List[float]],
        q_pos_start: int,
    ) -> List[List[float]]:
        # triton stub 未实现残差路径时回退 ref（保正确性）
        return causal_attn_queries(q, k, v, q_pos_start)


class CpuAttentionBackend:
    """纯 torch SDPA 后端，CPU/dev/test 验证路径（方案 A，对齐 SGLang ``torch_native``）。

    Tensor shape follows common decoder attention layout: [B, H, T, D]。
    生产 GPU 路径走 ``FlashAttn2Backend``（FA2 paged varlen）。
    """

    name = "cpu"

    def forward_tensors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = True,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must have shape [B, H, T, D]")
        if k.shape[1] != v.shape[1]:
            raise ValueError("k and v must have the same number of heads")
        if q.shape[1] != k.shape[1]:
            if q.shape[1] % k.shape[1] != 0:
                raise ValueError("q heads must be divisible by kv heads for GQA repeat")
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        causal = is_causal and attn_mask is None
        try:
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=scale,
            )
        except TypeError:
            if scale is not None:
                q = q * (scale * (q.shape[-1] ** 0.5))
            return torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
            )

    def forward_varlen(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_meta: AttentionMetadata,
        *,
        block_size: int = 8,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Ragged / paged 前向（方案 A，对齐 SGLang ``torch_native``）。

        入参：
            q        : [num_tokens, num_heads, head_dim] —— 本步 query token 拼接
            k_cache  : [total_slots, num_kv_heads, head_dim] —— 池 L0 arena 句柄（引擎只读）
            v_cache  : [total_slots, num_kv_heads, head_dim]
            attn_meta: AttentionMetadata —— block_table_tensor（block 级，按 req_order）、
                       query_start_loc、seq_lens_ordered
        返回：
            out      : [num_tokens, num_heads, head_dim]

        每请求：按 block_table 展开 slot → gather paged KV → SDPA。
        extend（q_len>1）：pad query 到 seq_len + is_causal=True + slice 出新 query 段。
        decode（q_len=1）：不 pad，直接 SDPA is_causal=False（单 token 对全 KV full-visible）。
        这两条正是 SGLang ``_run_sdpa_forward_extend`` / ``_run_sdpa_forward_decode`` 的等价写法，
        验证 lake CPU/dev 路径可行性（生产 GPU 由 FlashAttn2Backend 取代）。
        """
        if q.ndim != 3:
            raise ValueError("q must have shape [num_tokens, num_heads, head_dim]")
        if k_cache.ndim != 3 or v_cache.ndim != 3:
            raise ValueError("k_cache/v_cache must have shape [total_slots, num_kv_heads, head_dim]")
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        num_kv_heads = k_cache.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError("q heads must be divisible by kv heads for GQA repeat")
        repeat = num_heads // num_kv_heads
        out = torch.empty_like(q)
        qsl = attn_meta.query_start_loc
        seq_lens = attn_meta.seq_lens_ordered
        tables = attn_meta.block_table_tensor
        if len(qsl) - 1 != len(tables) or len(seq_lens) != len(tables):
            raise ValueError(
                "inconsistent metadata: query_start_loc/block_table_tensor/seq_lens_ordered"
            )
        for i in range(len(tables)):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens[i]
            if seq_len <= 0:
                raise ValueError(f"req {i}: seq_len={seq_len} but q_len={q_len}")
            # block 级 block_table → token 级 slot 列表（取前 seq_len 个 token）
            table = tables[i]
            slots = torch.empty(seq_len, dtype=torch.long, device=k_cache.device)
            for t in range(seq_len):
                blk = table[t // block_size]
                slots[t] = blk * block_size + (t % block_size)
            k_i = k_cache[slots]  # [seq_len, num_kv_heads, head_dim]
            v_i = v_cache[slots]
            if repeat != 1:
                k_i = k_i.repeat_interleave(repeat, dim=1)
                v_i = v_i.repeat_interleave(repeat, dim=1)
            q_i = q[q_start:q_end]  # [q_len, num_heads, head_dim]
            # SDPA 要 [B, H, T, D]：把 head 维提到前面（对齐 SGLang movedim(0, dim-2)）
            qh = q_i.movedim(0, 1).unsqueeze(0)  # [1, H, q_len, D]
            kh = k_i.movedim(0, 1).unsqueeze(0)  # [1, H, seq_len, D]
            vh = v_i.movedim(0, 1).unsqueeze(0)
            if q_len == 1:
                # decode：单 token 对全 KV full-visible，不 pad
                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    qh, kh, vh, is_causal=False, scale=scale
                )
            else:
                # extend：pad query 到 seq_len，is_causal=True，slice 出新 query 段
                prefix_len = seq_len - q_len
                padded = torch.zeros(
                    (1, num_heads, seq_len, head_dim),
                    dtype=qh.dtype,
                    device=qh.device,
                )
                padded[:, :, prefix_len:] = qh
                full_out = torch.nn.functional.scaled_dot_product_attention(
                    padded, kh, vh, is_causal=True, scale=scale
                )
                attn_out = full_out[:, :, prefix_len:prefix_len + q_len]
            # [1, H, q_len, D] → [q_len, H, D]
            out[q_start:q_end] = attn_out.squeeze(0).movedim(1, 0)
        return out


class FlashAttn2Backend:
    """GPU 生产后端（FlashAttention-2 paged varlen，上游 ``flash-attn`` 包）。

    入参约定同 ``CpuAttentionBackend.forward_varlen``，但走上游 ``flash-attn`` 的
    ``flash_attn_varlen_func``（CUDA kernel，paged KV via ``block_table``）。

    与 CPU 后端的差异：
    - GQA 原生支持（q heads 可 != kv heads，整除即可），无需 ``repeat_interleave``。
    - ``causal=True`` 统一处理 prefill / extend / decode：FA2 varlen 的 causal 是
      bottom-right 对齐（query 在序列末尾），三种模式 query 都在末尾，都成立——
      不必像 SGLang Triton 那样分 decode/extend 两个 kernel。
    - 上游 FA2 varlen **无 ``out=`` out-param**（返回 out），故**非入图路径**可用；
      CUDA graph capture（要固定地址输出 buffer）需改 vllm fork 或 Triton 后端。

    KV cache 布局：lake 池 L0 arena 是 flat ``[total_slots, Hkv, D]``，此处 view 成
    block-major ``[num_blocks, block_size, Hkv, D]`` 传给 FA2（``total_slots`` 须为
    ``block_size`` 整数倍）——与 ``flash_attn_with_kvcache`` 的 paged 布局一致。

    未验证（本环境无 GPU / 未装 ``flash-attn``，以下假设待 GPU 环境确认）：
    (1) 上游 FA2 varlen paged 是否接受 ``block_size < 256``：``flash_attn_with_kvcache``
        文档要求 ``page_block_size`` 为 256 倍数；varlen paged 路径约束待查。lake 当前
        ``block_size=8``——若上游要求 256 倍数，需改 block_size 或换 vllm fork（fork
        放宽了该约束，vLLM 用 block_size=16）。
    (2) paged 路径 k 用 4D block-major（此处取此）还是 3D flat——上游 docstring 写
        ``(total_k, nheads, headdim)``，但 paged 需从 k 形状读 block_size，故取 4D。
    (3) paged 路径用 ``cu_seqlens_k``（前缀和）表达 per-request KV 长度（上游公开 API
        无 ``seqused_k``；vllm fork 改用 ``seqused_k``，二者 API 不同）。
    """

    name = "fa2"

    def forward_varlen(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_meta: AttentionMetadata,
        *,
        block_size: int = 8,
        scale: float | None = None,
    ) -> torch.Tensor:
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                "FlashAttn2Backend 需要 `pip install flash-attn`（CUDA-only）。"
                " CPU 环境请用 CpuAttentionBackend。"
            ) from e

        if q.ndim != 3:
            raise ValueError("q must have shape [num_tokens, num_heads, head_dim]")
        if k_cache.ndim != 3 or v_cache.ndim != 3:
            raise ValueError("k_cache/v_cache must have shape [total_slots, num_kv_heads, head_dim]")
        total_slots = k_cache.shape[0]
        if total_slots % block_size != 0:
            raise ValueError(
                f"total_slots={total_slots} must be divisible by block_size={block_size}"
            )
        num_blocks = total_slots // block_size
        head_dim = q.shape[2]
        num_kv_heads = k_cache.shape[1]
        # flat [total_slots, Hkv, D] → block-major [num_blocks, block_size, Hkv, D]
        k = k_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
        v = v_cache.view(num_blocks, block_size, num_kv_heads, head_dim)

        B = attn_meta.num_reqs
        device = q.device
        cu_seqlens_q = torch.tensor(attn_meta.query_start_loc, dtype=torch.int32, device=device)
        # cu_seqlens_k = 前缀和(seq_lens_ordered)
        seq_lens = attn_meta.seq_lens_ordered
        cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=device)
        for i in range(B):
            cu_seqlens_k[i + 1] = cu_seqlens_k[i] + seq_lens[i]
        # block_table pad 到 2D [B, max_num_blocks]
        tables = attn_meta.block_table_tensor
        max_blocks = max((len(t) for t in tables), default=0) if B else 0
        block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
        for i, t in enumerate(tables):
            block_table[i, :len(t)] = torch.tensor(t, dtype=torch.int32, device=device)

        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=attn_meta.max_query_len,
            max_seqlen_k=attn_meta.max_seq_len,
            softmax_scale=scale if scale is not None else head_dim ** -0.5,
            causal=True,
            block_table=block_table,
        )


def build_attn_backend(name: str = "triton") -> AttentionBackend:
    if name == "ref":
        return RefAttentionBackend()
    if name == "cpu":
        return CpuAttentionBackend()
    if name == "fa2":
        return FlashAttn2Backend()
    return TritonAttentionBackend()
