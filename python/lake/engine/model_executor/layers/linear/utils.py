"""Linear 层工具函数。"""

from __future__ import annotations

from typing import Tuple

import torch


def divide(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"denominator must be > 0, got {denominator}")
    if numerator % denominator != 0:
        raise ValueError(f"{numerator} is not divisible by {denominator}")
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor, num_partitions: int
) -> Tuple[torch.Tensor, ...]:
    """沿最后一维均分（RowParallel 且 ``input_is_parallel=False`` 时用）。"""
    last = tensor.size(-1)
    if last % num_partitions != 0:
        raise ValueError(
            f"last dim {last} not divisible by num_partitions={num_partitions}"
        )
    return tuple(tensor.split(last // num_partitions, dim=-1))
