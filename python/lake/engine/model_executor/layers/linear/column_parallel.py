"""列并行 Linear（对齐 vLLM ``ColumnParallelLinear``）。"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lake.engine.distributed.communication_op import tensor_model_parallel_all_gather
from lake.engine.distributed.parallel_state import GroupCoordinator
from lake.engine.model_executor.layers.linear.base import LinearBase
from lake.engine.model_executor.layers.linear.utils import divide


class ColumnParallelLinearLayer(LinearBase):
    """列并行：``Y = X A``，``A`` 按列（输出维）切到各 rank。

    ``gather_output=True`` 时对本层 ``pg`` 做 all-gather。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
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
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            return_bias=return_bias,
            pg=pg,
            disable_tp=disable_tp,
            device=device,
        )
        self.gather_output = gather_output
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(
            self._empty(
                self.output_size_per_partition,
                self.input_size_per_partition,
            )
        )
        if bias:
            self.bias = nn.Parameter(self._empty(self.output_size_per_partition))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight.device.type == "meta":
            return
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """从完整（未切分）权重装载本 rank 分片；输出维 = dim 0。"""
        if loaded_weight.dim() == 0:
            loaded_weight = loaded_weight.reshape(1)
        if param is self.weight or (self.bias is not None and param is self.bias):
            shard = loaded_weight.narrow(
                0, self.tp_rank * param.shape[0], param.shape[0]
            )
            param.data.copy_(shard)
            return
        raise ValueError(f"unexpected param shape {tuple(param.shape)}")

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[nn.Parameter]]]:
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_, self.weight, bias)
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(
                output_parallel, dim=-1, group=self.pg
            )
        else:
            output = output_parallel
        return self._maybe_return(output, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"output_features={self.output_size_per_partition}, "
            f"bias={self.bias is not None}, tp_size={self.tp_size}, "
            f"gather_output={self.gather_output}, pg={self.pg.group_name}"
        )


ColumnParallelLinear = ColumnParallelLinearLayer
