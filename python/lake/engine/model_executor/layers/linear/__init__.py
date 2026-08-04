"""Megatron 风格并行 Linear 包（对齐 vLLM ``layers/linear.py``，按类分文件）。

- ``ColumnParallelLinearLayer``：列切 + 可选 all-gather（``pg``，默认 TP）
- ``RowParallelLinearLayer``：行切 + 可选 all-reduce（``pg``，默认 TP）
- ``ReplicatedLinearLayer``：全复制，无集体通信

参考：``vllm/model_executor/layers/linear.py``。
"""

from lake.engine.model_executor.layers.linear.base import LinearBase
from lake.engine.model_executor.layers.linear.column_parallel import (
    ColumnParallelLinear,
    ColumnParallelLinearLayer,
)
from lake.engine.model_executor.layers.linear.replicated import (
    ReplicatedLinear,
    ReplicatedLinearLayer,
)
from lake.engine.model_executor.layers.linear.row_parallel import (
    RowParallelLinear,
    RowParallelLinearLayer,
)
from lake.engine.model_executor.layers.linear.utils import (
    divide,
    split_tensor_along_last_dim,
)

__all__ = [
    "LinearBase",
    "ColumnParallelLinear",
    "ColumnParallelLinearLayer",
    "RowParallelLinear",
    "RowParallelLinearLayer",
    "ReplicatedLinear",
    "ReplicatedLinearLayer",
    "divide",
    "split_tensor_along_last_dim",
]
