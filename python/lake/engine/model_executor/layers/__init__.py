from lake.engine.model_executor.layers.attentions import AttentionMetadata, build_attn_metadata
from lake.engine.model_executor.layers.linear import (
    ColumnParallelLinear,
    ColumnParallelLinearLayer,
    MergedColumnParallelLinear,
    MergedColumnParallelLinearLayer,
    ReplicatedLinear,
    ReplicatedLinearLayer,
    RowParallelLinear,
    RowParallelLinearLayer,
)

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "ColumnParallelLinear",
    "ColumnParallelLinearLayer",
    "CpuAttentionBackend",
    "FlashAttn2Backend",
    "MergedColumnParallelLinear",
    "MergedColumnParallelLinearLayer",
    "RefAttentionBackend",
    "ReplicatedLinear",
    "ReplicatedLinearLayer",
    "RowParallelLinear",
    "RowParallelLinearLayer",
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
