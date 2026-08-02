"""参考后端：纯 Python list-based，早期原型兼容路径。"""

from __future__ import annotations

from typing import List

from lake.kernels.attn_ref import causal_attn, causal_attn_queries


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
