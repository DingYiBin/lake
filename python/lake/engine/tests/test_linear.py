"""Column / Row / Replicated linear（单进程；自定义 pg 可测分片形状）。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lake.engine.distributed.parallel_state import (
    GroupCoordinator,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
)
from lake.engine.config import ParallelConfig
from lake.engine.model_executor.layers.linear import (
    ColumnParallelLinearLayer,
    ReplicatedLinearLayer,
    RowParallelLinearLayer,
    divide,
)


def _fake_tp_group(rank: int, world_size: int) -> GroupCoordinator:
    """无 ProcessGroup 的 mesh 视图——只测分片形状 / 本地 GEMM，不测集体通信。"""
    return GroupCoordinator(
        ranks=list(range(world_size)),
        rank=rank,
        local_rank=rank,
        group_name=f"fake-tp-{world_size}",
    )


def test_divide() -> None:
    assert divide(8, 2) == 4


def test_column_row_tp1_matches_nn_linear() -> None:
    destroy_model_parallel()
    ensure_model_parallel_initialized(ParallelConfig())
    try:
        torch.manual_seed(0)
        col = ColumnParallelLinearLayer(4, 6, bias=True, gather_output=False)
        row = RowParallelLinearLayer(6, 3, bias=True, reduce_results=True)
        x = torch.randn(2, 4)
        y = col(x)
        assert isinstance(y, torch.Tensor)
        assert y.shape == (2, 6)
        z = row(y)
        assert z.shape == (2, 3)
        # 对照 F.linear
        assert torch.allclose(y, F.linear(x, col.weight, col.bias))
    finally:
        destroy_model_parallel()


def test_column_shards_output_dim() -> None:
    g0 = _fake_tp_group(0, 2)
    g1 = _fake_tp_group(1, 2)
    col0 = ColumnParallelLinearLayer(4, 8, bias=False, pg=g0)
    col1 = ColumnParallelLinearLayer(4, 8, bias=False, pg=g1)
    assert col0.weight.shape == (4, 4)
    assert col1.weight.shape == (4, 4)
    assert col0.tp_rank == 0 and col1.tp_rank == 1

    full = torch.randn(8, 4)
    col0.weight_loader(col0.weight, full)
    col1.weight_loader(col1.weight, full)
    assert torch.equal(col0.weight, full[:4])
    assert torch.equal(col1.weight, full[4:])

    x = torch.randn(3, 4)
    y0 = col0(x)
    y1 = col1(x)
    assert y0.shape == (3, 4)
    assert y1.shape == (3, 4)
    # 本地分片 GEMM 拼起来应等于完整线性（无 gather 时由调用方拼）
    y_full = F.linear(x, full, None)
    assert torch.allclose(torch.cat([y0, y1], dim=-1), y_full)


def test_row_shards_input_dim_and_sums() -> None:
    g0 = _fake_tp_group(0, 2)
    g1 = _fake_tp_group(1, 2)
    # reduce_results 需要 device_group；tp>1 无 PG 时关掉 reduce，手动求和对照
    row0 = RowParallelLinearLayer(8, 3, bias=False, pg=g0, reduce_results=False)
    row1 = RowParallelLinearLayer(8, 3, bias=False, pg=g1, reduce_results=False)
    assert row0.weight.shape == (3, 4)

    full = torch.randn(3, 8)
    row0.weight_loader(row0.weight, full)
    row1.weight_loader(row1.weight, full)
    x = torch.randn(2, 8)
    x0, x1 = x[:, :4], x[:, 4:]
    y = row0(x0) + row1(x1)
    assert torch.allclose(y, F.linear(x, full, None), atol=1e-5)


def test_custom_pg_not_default_tp() -> None:
    destroy_model_parallel()
    ensure_model_parallel_initialized(ParallelConfig())  # default tp=1
    try:
        custom = _fake_tp_group(0, 4)
        col = ColumnParallelLinearLayer(8, 16, bias=False, pg=custom)
        assert col.tp_size == 4
        assert col.pg.group_name == "fake-tp-4"
        assert col.weight.shape == (4, 8)
    finally:
        destroy_model_parallel()


def test_disable_tp_ignores_group_size() -> None:
    custom = _fake_tp_group(0, 4)
    col = ColumnParallelLinearLayer(8, 16, bias=False, pg=custom, disable_tp=True)
    assert col.tp_size == 1
    assert col.weight.shape == (16, 8)


def test_replicated_full_weight_no_shard() -> None:
    custom = _fake_tp_group(0, 4)
    layer = ReplicatedLinearLayer(5, 7, bias=True, pg=custom)
    assert layer.weight.shape == (7, 5)
    assert layer.bias is not None and layer.bias.shape == (7,)
    # 即便传入多卡 pg，也不切分、不用集体通信
    assert layer.tp_size == 1

    full_w = torch.randn(7, 5)
    full_b = torch.randn(7)
    layer.weight_loader(layer.weight, full_w)
    layer.weight_loader(layer.bias, full_b)
    x = torch.randn(3, 5)
    y = layer(x)
    assert isinstance(y, torch.Tensor)
    assert torch.allclose(y, F.linear(x, full_w, full_b))
