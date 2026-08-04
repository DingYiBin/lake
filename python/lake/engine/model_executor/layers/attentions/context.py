"""Forward context：把本步 ``AttentionMetadata`` + per-layer KV arena 句柄注入
模型 forward，使 ``Qwen3Attention`` 等层不必经模型 API 显式传递。

对齐 vLLM ``set_forward_context`` / SGLang forward context：runner 在调模型
forward 前设上下文，各 attention 层按 ``layer_idx`` 取自己那层 ``k_cache``/``v_cache``。
lake 的 KV arena 归存储池（方案 Z）；本上下文只承载句柄引用，不转移所有权。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class ForwardContext:
    """单步 forward 上下文。``kv_caches`` 为 per-layer ``(k_cache, v_cache)`` 列表。"""

    attn_meta: Optional[object] = None
    kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None


_current: Optional[ForwardContext] = None


def get_forward_context() -> Optional[ForwardContext]:
    return _current


@contextlib.contextmanager
def forward_context(
    attn_meta: object,
    kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
):
    global _current
    prev = _current
    _current = ForwardContext(attn_meta=attn_meta, kv_caches=kv_caches)
    try:
        yield
    finally:
        _current = prev
