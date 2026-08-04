"""并行 Linear 基类（对齐 vLLM ``LinearBase``）。"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from lake.engine.distributed.parallel_state import (
    GroupCoordinator,
    resolve_comm_group,
)


class LinearBase(nn.Module):
    """并行 Linear 公共字段（无量化路径；GEMM 走 ``F.linear``）。

    ``pg``：通讯域；``None`` 时取默认 TP（见 ``resolve_comm_group``）。
    ``disable_tp=True`` 时强制单卡组（权重不切分语义由子类决定）。
    """

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
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        self.device = device
        self.params_dtype = (
            params_dtype if params_dtype is not None else torch.get_default_dtype()
        )
        if disable_tp:
            self.pg = GroupCoordinator(
                ranks=[0], rank=0, local_rank=0, group_name="tp-disabled"
            )
        else:
            self.pg = resolve_comm_group(pg)
        self.tp_size = self.pg.world_size
        self.tp_rank = self.pg.rank_in_group

    def _empty(self, *shape: int) -> torch.Tensor:
        return torch.empty(*shape, dtype=self.params_dtype, device=self.device)

    def _maybe_return(
        self, output: torch.Tensor, bias: Optional[nn.Parameter]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[nn.Parameter]]]:
        if not self.return_bias:
            return output
        output_bias = bias if self.skip_bias_add else None
        return output, output_bias
