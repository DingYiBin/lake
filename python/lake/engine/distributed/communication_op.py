"""TP 集体通信薄封装（对齐 vLLM ``communication_op``）。

与 vLLM 差异：接受可选 ``group``；未传时用默认 TP 组。这样
Column/Row parallel linear 可自行指定通讯域（例如未来 attn_tp / EP）。
"""

from __future__ import annotations

from typing import Optional

import torch

from lake.engine.distributed.parallel_state import (
    GroupCoordinator,
    get_tp_group,
    model_parallel_is_initialized,
    resolve_comm_group,
)


def _default_group(group: Optional[GroupCoordinator]) -> GroupCoordinator:
    if group is not None:
        return group
    if model_parallel_is_initialized():
        return get_tp_group()
    return resolve_comm_group(None)


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
    group: Optional[GroupCoordinator] = None,
) -> torch.Tensor:
    """All-reduce across the given (or default TP) group."""
    return _default_group(group).all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor,
    dim: int = -1,
    group: Optional[GroupCoordinator] = None,
) -> torch.Tensor:
    """All-gather across the given (or default TP) group."""
    return _default_group(group).all_gather(input_, dim=dim)
