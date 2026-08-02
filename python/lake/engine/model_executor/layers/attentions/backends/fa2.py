"""FA2 后端：上游 ``flash-attn`` 包 paged varlen，GPU 生产路径（非入图）。"""

from __future__ import annotations

import torch

from lake.engine.model_executor.layers.attentions.attention_metadata import (
    AttentionMetadata,
)


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
