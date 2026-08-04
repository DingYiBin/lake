"""融合列并行 Linear（对齐 vLLM ``MergedColumnParallelLinear``）。

把多个逻辑输出段（如 gate + up）拼成一张列切权重；加载时按段分别
对 TP 取 shard，再写入 fused 参数的对应区间——与「整段 fused 再按 TP
切 contiguous」不同。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from lake.engine.distributed.parallel_state import GroupCoordinator
from lake.engine.model_executor.layers.linear.column_parallel import (
    ColumnParallelLinearLayer,
)
from lake.engine.model_executor.layers.linear.utils import divide


class MergedColumnParallelLinearLayer(ColumnParallelLinearLayer):
    """Packed column-parallel：``output_sizes`` 各段沿输出维拼接。

    例：FFN ``gate_up_proj`` → ``output_sizes=[intermediate, intermediate]``。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        *,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        pg: Optional[GroupCoordinator] = None,
        disable_tp: bool = False,
        device: Optional[torch.device | str] = None,
    ) -> None:
        sizes = list(output_sizes)
        if not sizes:
            raise ValueError("output_sizes must be non-empty")
        if any(s < 1 for s in sizes):
            raise ValueError(f"output_sizes must be positive, got {sizes}")

        self.output_sizes: List[int] = sizes
        super().__init__(
            input_size,
            sum(sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            return_bias=return_bias,
            pg=pg,
            disable_tp=disable_tp,
            device=device,
        )
        # 每段必须能被 tp 整除（构造 Column 时已保证 sum 可除；再钉每段）
        for s in self.output_sizes:
            divide(s, self.tp_size)
        self.output_partition_sizes = [divide(s, self.tp_size) for s in self.output_sizes]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ) -> None:
        """装载权重。

        - ``loaded_shard_id is None``：磁盘上已是 fused（如 ``gate_up_proj``），
          按各逻辑段切开后分别装载。
        - ``loaded_shard_id: int``：装某一段（0=gate，1=up…），对该段做 TP narrow
          后写入 fused 参数中对应本地区间。
        """
        if loaded_weight.dim() == 0:
            loaded_weight = loaded_weight.reshape(1)

        if loaded_shard_id is None:
            offset = 0
            for shard_id, shard_size in enumerate(self.output_sizes):
                piece = loaded_weight.narrow(0, offset, shard_size)
                self.weight_loader(param, piece, loaded_shard_id=shard_id)
                offset += shard_size
            return

        if not (0 <= loaded_shard_id < len(self.output_sizes)):
            raise ValueError(
                f"loaded_shard_id={loaded_shard_id} out of range "
                f"[0, {len(self.output_sizes)})"
            )

        # fused 参数内本 rank 上该段的偏移/宽度（已按 TP 缩小）
        local_offset = sum(self.output_partition_sizes[:loaded_shard_id])
        local_size = self.output_partition_sizes[loaded_shard_id]
        dest = param.data.narrow(0, local_offset, local_size)

        # 完整单段权重 → 取本 tp_rank 分片
        src = loaded_weight.narrow(0, self.tp_rank * local_size, local_size)
        if dest.shape != src.shape:
            raise ValueError(
                f"merged shard {loaded_shard_id}: dest {tuple(dest.shape)} "
                f"!= src {tuple(src.shape)}"
            )
        dest.copy_(src)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"output_sizes={self.output_sizes}, "
            f"output_features={self.output_size_per_partition}, "
            f"bias={self.bias is not None}, tp_size={self.tp_size}, "
            f"gather_output={self.gather_output}, pg={self.pg.group_name}"
        )


MergedColumnParallelLinear = MergedColumnParallelLinearLayer
