from lake.engine.model_executor.layers.attentions.attention_metadata import (
    AttentionMetadata,
    build_attn_metadata,
)

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "CpuAttentionBackend",
    "FlashAttn2Backend",
    "RefAttentionBackend",
    "build_attn_backend",
    "build_attn_metadata",
]


def __getattr__(name: str):
    # 懒委托到 backends 子包：import attentions 包不拉入任何具体后端/kernel
    if name in {
        "AttentionBackend",
        "CpuAttentionBackend",
        "FlashAttn2Backend",
        "RefAttentionBackend",
        "TritonAttentionBackend",
        "build_attn_backend",
    }:
        from lake.engine.model_executor.layers.attentions import backends

        return getattr(backends, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
