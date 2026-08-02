from lake.engine.model_executor.layers.attentions import AttentionMetadata, build_attn_metadata

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
    if name in {
        "AttentionBackend",
        "CpuAttentionBackend",
        "FlashAttn2Backend",
        "RefAttentionBackend",
        "build_attn_backend",
    }:
        from lake.engine.model_executor.layers import attentions

        return getattr(attentions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
