"""并行规模与分布式 bootstrap 配置（对齐 vLLM ParallelConfig 子集）。

第一版只覆盖 lake 近期要用的 TP / DP / PP(预留) 与建组所需地址；
EP / EPLB / Ray / DBO 等后置，不在此展开。

参考：``vllm/config/parallel.py::ParallelConfig``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from lake.engine.config.utils import env_bool, env_int


@dataclass
class ParallelConfig:
    """分布式执行规模。mesh 约定：``DP × PP × TP``（对齐 vLLM，PCP=1）。"""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1  # 预留；当前实现按 1 对待
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    # torch.distributed bootstrap（world_size_across_dp > 1 时用）
    master_addr: str = "127.0.0.1"
    master_port: int = 29501
    # DP 协调口（跨 DP 组 init / 后续 sync；与 master_port 分开避免冲突）
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_master_port: int = 29500
    # 本进程在 torch world 内的全局 rank / 本机 local rank；-1 = 未绑定
    rank: int = -1
    local_rank: int = -1
    disable_custom_all_reduce: bool = False

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        ):
            val = getattr(self, name)
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")
        if not (0 <= self.data_parallel_rank < self.data_parallel_size):
            raise ValueError(
                f"data_parallel_rank={self.data_parallel_rank} out of range "
                f"[0, {self.data_parallel_size})"
            )
        if self.master_port <= 0 or self.data_parallel_master_port <= 0:
            raise ValueError("master ports must be > 0")

    @property
    def world_size(self) -> int:
        """单 DP 副本内的 executor world：``TP × PP``。"""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def world_size_across_dp(self) -> int:
        """含 DP 的全局 torch world：``DP × TP × PP``。"""
        return self.world_size * self.data_parallel_size

    @property
    def is_distributed(self) -> bool:
        return self.world_size_across_dp > 1

    def resolve_rank(self) -> int:
        """全局 rank：显式 ``rank`` 优先，否则按 ``dp_rank * world + local`` 推。

        local 侧默认取 ``local_rank``（>=0）否则 0——单进程 TP=1 时足够；
        多卡由 launcher 注入 ``rank`` / ``LOCAL_RANK``。
        """
        if self.rank >= 0:
            return self.rank
        local = self.local_rank if self.local_rank >= 0 else 0
        if local >= self.world_size:
            raise ValueError(
                f"local_rank={local} >= world_size={self.world_size} "
                "(TP×PP within one DP replica)"
            )
        return self.data_parallel_rank * self.world_size + local

    def resolve_local_rank(self) -> int:
        if self.local_rank >= 0:
            return self.local_rank
        if self.rank >= 0:
            return self.rank % max(self.world_size, 1)
        return 0

    @classmethod
    def from_env(cls) -> "ParallelConfig":
        """环境变量（对齐 vLLM 语义，前缀 LAKE_）。

        LAKE_TP_SIZE / LAKE_PP_SIZE / LAKE_DP_SIZE / LAKE_DP_RANK
        LAKE_MASTER_ADDR / LAKE_MASTER_PORT
        LAKE_DP_MASTER_IP / LAKE_DP_MASTER_PORT
        LAKE_RANK / LAKE_LOCAL_RANK（或标准 RANK / LOCAL_RANK）
        LAKE_DISABLE_CUSTOM_ALL_REDUCE
        """
        rank = env_int("LAKE_RANK", -1)
        if rank < 0:
            rank = env_int("RANK", -1)
        local_rank = env_int("LAKE_LOCAL_RANK", -1)
        if local_rank < 0:
            local_rank = env_int("LOCAL_RANK", -1)
        return cls(
            tensor_parallel_size=env_int("LAKE_TP_SIZE", 1),
            pipeline_parallel_size=env_int("LAKE_PP_SIZE", 1),
            data_parallel_size=env_int("LAKE_DP_SIZE", 1),
            data_parallel_rank=env_int("LAKE_DP_RANK", 0),
            master_addr=os.environ.get("LAKE_MASTER_ADDR", "127.0.0.1").strip()
            or "127.0.0.1",
            master_port=env_int("LAKE_MASTER_PORT", 29501),
            data_parallel_master_ip=os.environ.get(
                "LAKE_DP_MASTER_IP", "127.0.0.1"
            ).strip()
            or "127.0.0.1",
            data_parallel_master_port=env_int("LAKE_DP_MASTER_PORT", 29500),
            rank=rank,
            local_rank=local_rank,
            disable_custom_all_reduce=env_bool(
                "LAKE_DISABLE_CUSTOM_ALL_REDUCE", False
            ),
        )
