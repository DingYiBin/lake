"""分布式通讯域（parallel_state）。

``ParallelConfig`` 权威定义在 ``lake.engine.config``；此处保留 re-export 以便
``from lake.engine.distributed import ParallelConfig`` 仍可用。
"""

from lake.engine.config.parallel import ParallelConfig
from lake.engine.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from lake.engine.distributed.parallel_state import (
    GroupCoordinator,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_dp_group,
    get_parallel_config,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
    resolve_comm_group,
)

__all__ = [
    "ParallelConfig",
    "GroupCoordinator",
    "destroy_model_parallel",
    "ensure_model_parallel_initialized",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_dp_group",
    "get_parallel_config",
    "get_pipeline_model_parallel_rank",
    "get_pipeline_model_parallel_world_size",
    "get_pp_group",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "get_tp_group",
    "get_world_group",
    "init_distributed_environment",
    "initialize_model_parallel",
    "model_parallel_is_initialized",
    "resolve_comm_group",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_all_reduce",
]
