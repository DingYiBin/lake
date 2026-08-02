"""Attention 后端注册表（对照 vLLM ``AttentionBackendEnum`` / SGLang ``ATTENTION_BACKENDS``）。

按名实例化后端，**懒导入**——import 本模块不拉入任何具体后端或 kernel，
仅 ``build_attn_backend(name)`` 被调用时才 import 对应后端文件。
"""

from __future__ import annotations

from lake.engine.model_executor.layers.attentions.backends.base import AttentionBackend


def build_attn_backend(name: str = "triton") -> AttentionBackend:
    if name == "ref":
        from lake.engine.model_executor.layers.attentions.backends.ref import (
            RefAttentionBackend,
        )

        return RefAttentionBackend()
    if name == "cpu":
        from lake.engine.model_executor.layers.attentions.backends.cpu import (
            CpuAttentionBackend,
        )

        return CpuAttentionBackend()
    if name == "fa2":
        from lake.engine.model_executor.layers.attentions.backends.fa2 import (
            FlashAttn2Backend,
        )

        return FlashAttn2Backend()
    from lake.engine.model_executor.layers.attentions.backends.triton import (
        TritonAttentionBackend,
    )

    return TritonAttentionBackend()
