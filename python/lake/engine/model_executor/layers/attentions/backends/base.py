"""Attention 后端协议（对照 vLLM ``AttentionBackend`` ABC）。

具体后端各自独立成文件（``ref.py`` / ``triton.py`` / ``cpu.py`` / ``fa2.py``），
由 ``registry.py::build_attn_backend`` 按名实例化。模型层只持注入的 backend 句柄，
不 import 任何具体后端（见 ``qwen3.py``）。
"""

from __future__ import annotations

from typing import List, Protocol


class AttentionBackend(Protocol):
    """attention 后端协议（早期 list-based 接口；tensor 路径见各后端 ``forward_tensors``/``forward_varlen``）。"""

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
