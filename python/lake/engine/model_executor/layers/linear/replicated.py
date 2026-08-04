"""全复制 Linear（对齐 vLLM ``ReplicatedLinear``）。

不使用任何通讯域：全量权重、forward 无 collective。
``pg`` / ``disable_tp`` 仅保留 API 对齐，对权重形状与 forward 无实质作用
（vLLM 文档亦写 ``disable_tp: Take no effect for replicated linear layers``）。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lake.engine.distributed.parallel_state import GroupCoordinator
from lake.engine.model_executor.layers.linear.base import LinearBase


class ReplicatedLinearLayer(LinearBase):
    """各 rank 持有完整 ``W``；本地 GEMM，无 all-reduce / all-gather。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        pg: Optional[GroupCoordinator] = None,
        disable_tp: bool = False,
    ) -> None:
        # 权重始终全量；强制单卡组语义，避免误用 tp_size 切分。
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            return_bias=return_bias,
            pg=pg,
            disable_tp=True,
        )
        # 保留调用方传入值，便于日志/对照（与 vLLM「disable_tp 无效果」一致）
        self.disable_tp = disable_tp
        self.output_partition_sizes = [output_size]

        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.input_size, dtype=self.params_dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.dim() == 0:
            loaded_weight = loaded_weight.reshape(1)
        if param.size() != loaded_weight.size():
            raise ValueError(
                f"Tried to load weights of size {tuple(loaded_weight.size())} "
                f"to a parameter of size {tuple(param.size())}"
            )
        param.data.copy_(loaded_weight)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[nn.Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        output = F.linear(x, self.weight, bias)
        return self._maybe_return(output, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"output_features={self.output_size}, "
            f"bias={self.bias is not None}"
        )


ReplicatedLinear = ReplicatedLinearLayer
