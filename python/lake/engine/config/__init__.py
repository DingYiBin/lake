"""计算层配置（对齐 vLLM ``vllm/config/``）。

权威路径：``lake.engine.config``。旧入口 ``lake.runtime.role`` /
``lake.engine.distributed.ParallelConfig`` 仅作兼容 re-export。
"""

from lake.engine.config.parallel import ParallelConfig
from lake.engine.config.role import RoleConfig, WorkerRole

__all__ = [
    "ParallelConfig",
    "RoleConfig",
    "WorkerRole",
]
