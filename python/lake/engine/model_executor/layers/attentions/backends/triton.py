"""Triton 后端：早期 list-based 原型，残差路径回退 ref 保正确性。"""

from __future__ import annotations

from typing import List

from lake.kernels.attn_ref import causal_attn_queries
from lake.kernels.attn_triton import causal_attn_triton


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
