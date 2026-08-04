"""行并行 Linear（对齐 vLLM ``RowParallelLinear``）。"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from lake.engine.distributed.communication_op import tensor_model_parallel_all_reduce
from lake.engine.distributed.parallel_state import GroupCoordinator
from lake.engine.model_executor.layers.linear.base import LinearBase
from lake.engine.model_executor.layers.linear.utils import (
    divide,
    split_tensor_along_last_dim,
)


class RowParallelLinearLayer(LinearBase):
    """行并行：``Y = X A``，``A`` 按行（输入维）切到各 rank。

    ``reduce_results=True`` 时对本层 ``pg`` 做 all-reduce。
    bias 仅在 ``tp_rank==0`` 融入 GEMM，避免重复加。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool = True,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_bias: bool = False,
        pg: Optional[GroupCoordinator] = None,
        disable_tp: bool = False,
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
        )
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        if not reduce_results and bias and not skip_bias_add:
            raise ValueError(
                "When not reducing results, adding bias can lead to incorrect results"
            )

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size_per_partition,
                dtype=self.params_dtype,
            )
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
        """从完整权重装载本 rank 分片；权重输入维 = dim 1。"""
        if loaded_weight.dim() == 0:
            loaded_weight = loaded_weight.reshape(1)
        if param is self.weight:
            shard = loaded_weight.narrow(
                1, self.tp_rank * param.shape[1], param.shape[1]
            )
            param.data.copy_(shard)
            return
        if self.bias is not None and param is self.bias:
            param.data.copy_(loaded_weight)
            return
        raise ValueError(f"unexpected param shape {tuple(param.shape)}")

    def forward(
        self, input_: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[nn.Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            chunks = split_tensor_along_last_dim(input_, self.tp_size)
            input_parallel = chunks[self.tp_rank].contiguous()

        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = F.linear(input_parallel, self.weight, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel, group=self.pg)
        else:
            output = output_parallel
        return self._maybe_return(output, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size_per_partition}, "
            f"output_features={self.output_size}, "
            f"bias={self.bias is not None}, tp_size={self.tp_size}, "
            f"reduce_results={self.reduce_results}, pg={self.pg.group_name}"
        )


RowParallelLinear = RowParallelLinearLayer
