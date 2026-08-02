"""Attention 后端选择（对照 vLLM AttentionBackend；agent 出表、runner 出其余 metadata 留 D4）。"""

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


class TorchAttentionBackend:
    """Pure Torch tensor backend for Qwen3/Torch model skeletons.

    Tensor shape follows common decoder attention layout: [B, H, T, D].
    """

    name = "torch"

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
        验证 lake CPU/dev 路径可行性（生产 GPU 由 Triton paged varlen kernel 取代）。
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


def build_attn_backend(name: str = "triton") -> AttentionBackend:
    if name == "ref":
        return RefAttentionBackend()
    if name == "torch":
        return TorchAttentionBackend()
    return TritonAttentionBackend()
