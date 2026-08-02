"""Attention 后端集合（对照 vLLM ``v1/attention/backends/``）。

后端矩阵：
- ``CpuAttentionBackend``（``cpu.py``）：纯 torch SDPA，CPU/dev/test 验证路径（方案 A，对齐 SGLang ``torch_native``）。
- ``FlashAttn2Backend``（``fa2.py``）：上游 ``flash-attn`` 包 FA2 paged varlen，GPU 生产路径（非入图）。
- ``RefAttentionBackend``（``ref.py``）/ ``TritonAttentionBackend``（``triton.py``）：早期 list-based 原型，保留兼容。

基类 ``AttentionBackend`` 在 ``base.py``；按名实例化走 ``registry.py::build_attn_backend``（懒导入）。
"""

from __future__ import annotations

from lake.engine.model_executor.layers.attentions.backends.base import AttentionBackend
from lake.engine.model_executor.layers.attentions.backends.registry import build_attn_backend

__all__ = [
    "AttentionBackend",
    "CpuAttentionBackend",
    "FlashAttn2Backend",
    "RefAttentionBackend",
    "TritonAttentionBackend",
    "build_attn_backend",
]


def __getattr__(name: str):
    # 懒导入具体后端：import backends 包不拉入 kernel/flash-attn
    _map = {
        "CpuAttentionBackend": "cpu",
        "FlashAttn2Backend": "fa2",
        "RefAttentionBackend": "ref",
        "TritonAttentionBackend": "triton",
    }
    if name in _map:
        import importlib

        mod = importlib.import_module(
            f"lake.engine.model_executor.layers.attentions.backends.{_map[name]}"
        )
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
