"""ParallelConfig + mesh 切组（无 NCCL）。"""

from __future__ import annotations

import os

import pytest

from lake.engine.config import ParallelConfig, RoleConfig
from lake.engine.distributed import (
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_data_parallel_rank,
    get_dp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tp_group,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from lake.engine.distributed.parallel_state import (
    dp_group_ranks,
    pp_group_ranks,
    tp_group_ranks,
)


def test_parallel_config_world_sizes() -> None:
    cfg = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2, data_parallel_size=2)
    assert cfg.world_size == 4
    assert cfg.world_size_across_dp == 8
    assert cfg.is_distributed is True


def test_parallel_config_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match="data_parallel_rank"):
        ParallelConfig(data_parallel_size=2, data_parallel_rank=2)


def test_resolve_rank_from_dp_and_local() -> None:
    cfg = ParallelConfig(
        tensor_parallel_size=2,
        data_parallel_size=2,
        data_parallel_rank=1,
        local_rank=1,
    )
    # dp=1, world=2 → rank = 1*2 + 1 = 3
    assert cfg.resolve_rank() == 3


def test_mesh_tp_pp_dp_groups_dp2_pp1_tp2() -> None:
    # ranks: dp0=[0,1], dp1=[2,3]
    assert tp_group_ranks(2, 1, 2) == [[0, 1], [2, 3]]
    assert dp_group_ranks(2, 1, 2) == [[0, 2], [1, 3]]
    assert pp_group_ranks(2, 1, 2) == [[0], [1], [2], [3]]


def test_mesh_with_pp() -> None:
    # DP=1 PP=2 TP=2 → ranks 0..3 as [pp0:[0,1], pp1:[2,3]]
    assert tp_group_ranks(1, 2, 2) == [[0, 1], [2, 3]]
    assert pp_group_ranks(1, 2, 2) == [[0, 2], [1, 3]]
    assert dp_group_ranks(1, 2, 2) == [[0], [1], [2], [3]]


def test_initialize_single_process_groups() -> None:
    destroy_model_parallel()
    try:
        ensure_model_parallel_initialized(ParallelConfig())
        assert model_parallel_is_initialized()
        assert get_tp_group().world_size == 1
        assert get_dp_group().world_size == 1
        assert get_pp_group().world_size == 1
        assert get_tensor_model_parallel_rank() == 0
        assert get_data_parallel_rank() == 0
        assert get_tp_group().device_group is None
    finally:
        destroy_model_parallel()


def test_initialize_mesh_views_without_dist() -> None:
    """world>1 但未 init dist 时 initialize_model_parallel 应失败。"""
    destroy_model_parallel()
    cfg = ParallelConfig(tensor_parallel_size=2, rank=0, local_rank=0)
    with pytest.raises(RuntimeError, match="init_distributed_environment"):
        initialize_model_parallel(cfg)
    assert not model_parallel_is_initialized()


def test_role_config_embeds_parallel_from_env() -> None:
    keys = ["LAKE_TP_SIZE", "LAKE_DP_SIZE", "LAKE_DP_RANK"]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["LAKE_TP_SIZE"] = "2"
        os.environ["LAKE_DP_SIZE"] = "2"
        os.environ["LAKE_DP_RANK"] = "1"
        cfg = RoleConfig.from_env()
        assert cfg.parallel.tensor_parallel_size == 2
        assert cfg.parallel.data_parallel_size == 2
        assert cfg.parallel.data_parallel_rank == 1
        assert cfg.parallel.world_size_across_dp == 4
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
