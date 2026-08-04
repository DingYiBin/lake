"""通讯域骨架：按 ``DP × PP × TP`` mesh 切出 TP/PP/DP 组。

对齐 vLLM ``parallel_state.initialize_model_parallel`` 的 reshape 约定
（PCP 固定为 1）。``world_size_across_dp == 1`` 时不碰 ``torch.distributed``，
便于单卡/单测。

真多卡时：先 ``init_distributed_environment``，再 ``initialize_model_parallel``。
ProcessGroup 创建在 ``world > 1`` 且 dist 已 init 时发生；否则只保留 rank 列表。

参考：
- ``vllm/distributed/parallel_state.py::GroupCoordinator``
- ``vllm/distributed/parallel_state.py::initialize_model_parallel``
- ``vllm/distributed/parallel_state.py::init_distributed_environment``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from lake.engine.config.parallel import ParallelConfig

# ---------------------------------------------------------------------------
# Mesh helpers（纯函数，可单测、无需 NCCL）
# ---------------------------------------------------------------------------


def build_mesh_ranks(
    *,
    data_parallel_size: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
) -> List[List[List[int]]]:
    """返回形状 ``[dp][pp][tp]`` 的全局 rank 网格。

    全局 layout（与 vLLM 一致，PCP=1）::

        rank = ((dp * pp_size) + pp) * tp_size + tp
    """
    if min(data_parallel_size, pipeline_parallel_size, tensor_parallel_size) < 1:
        raise ValueError("parallel sizes must be >= 1")
    mesh: List[List[List[int]]] = []
    for dp in range(data_parallel_size):
        pp_rows: List[List[int]] = []
        for pp in range(pipeline_parallel_size):
            row = [
                ((dp * pipeline_parallel_size) + pp) * tensor_parallel_size + tp
                for tp in range(tensor_parallel_size)
            ]
            pp_rows.append(row)
        mesh.append(pp_rows)
    return mesh


def tp_group_ranks(
    data_parallel_size: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
) -> List[List[int]]:
    """每个 ``(dp, pp)`` 一条 TP 组。"""
    mesh = build_mesh_ranks(
        data_parallel_size=data_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
    )
    return [mesh[dp][pp] for dp in range(data_parallel_size) for pp in range(pipeline_parallel_size)]


def pp_group_ranks(
    data_parallel_size: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
) -> List[List[int]]:
    """每个 ``(dp, tp)`` 一条 PP 组。"""
    mesh = build_mesh_ranks(
        data_parallel_size=data_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
    )
    groups: List[List[int]] = []
    for dp in range(data_parallel_size):
        for tp in range(tensor_parallel_size):
            groups.append([mesh[dp][pp][tp] for pp in range(pipeline_parallel_size)])
    return groups


def dp_group_ranks(
    data_parallel_size: int,
    pipeline_parallel_size: int,
    tensor_parallel_size: int,
) -> List[List[int]]:
    """每个 ``(pp, tp)`` 一条 DP 组。"""
    mesh = build_mesh_ranks(
        data_parallel_size=data_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
    )
    groups: List[List[int]] = []
    for pp in range(pipeline_parallel_size):
        for tp in range(tensor_parallel_size):
            groups.append([mesh[dp][pp][tp] for dp in range(data_parallel_size)])
    return groups


def find_group_containing(groups: Sequence[Sequence[int]], rank: int) -> List[int]:
    for g in groups:
        if rank in g:
            return list(g)
    raise ValueError(f"rank {rank} not found in any group")


# ---------------------------------------------------------------------------
# GroupCoordinator（瘦骨架：ranks + 可选 ProcessGroup）
# ---------------------------------------------------------------------------


@dataclass
class GroupCoordinator:
    """通信组视图。collective 走 ``device_group``（有则 ``torch.distributed``）。

    ``world_size == 1`` 时 all_reduce / all_gather 为恒等，便于单卡与单测。
    自定义 AR / PyNCCL 后续挂此对象，接口保持不变。
    """

    ranks: List[int]
    rank: int  # global rank
    local_rank: int
    group_name: str
    device_group: object | None = None  # torch.distributed.ProcessGroup | None
    cpu_group: object | None = None

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    @property
    def rank_in_group(self) -> int:
        return self.ranks.index(self.rank)

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:
        return self.rank_in_group == self.world_size - 1

    def all_reduce(self, input_: "torch.Tensor") -> "torch.Tensor":
        """Out-of-place all-reduce。``world_size==1`` 直接返回。"""
        import torch
        import torch.distributed as dist

        if self.world_size == 1:
            return input_
        if self.device_group is None:
            raise RuntimeError(
                f"GroupCoordinator({self.group_name}) has no device_group for all_reduce "
                f"(world_size={self.world_size})"
            )
        out = input_.clone()
        dist.all_reduce(out, group=self.device_group)
        return out

    def all_gather(self, input_: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
        """沿 ``dim`` all-gather 后 cat。``world_size==1`` 直接返回。"""
        import torch
        import torch.distributed as dist

        if self.world_size == 1:
            return input_
        if self.device_group is None:
            raise RuntimeError(
                f"GroupCoordinator({self.group_name}) has no device_group for all_gather "
                f"(world_size={self.world_size})"
            )
        if dim < 0:
            dim += input_.dim()
        tensor_list = [torch.empty_like(input_) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, input_.contiguous(), group=self.device_group)
        return torch.cat(tensor_list, dim=dim)


def resolve_comm_group(
    pg: Optional[GroupCoordinator] = None,
) -> GroupCoordinator:
    """解析 linear 等层使用的通讯域：显式 ``pg`` → 默认 TP → 单卡 fallback。

    单卡 fallback 仅用于尚未 ``initialize_model_parallel`` 的构造/单测；
    多卡务必先建组或显式传入 ``pg``。
    """
    if pg is not None:
        return pg
    if model_parallel_is_initialized():
        return get_tp_group()
    return GroupCoordinator(
        ranks=[0],
        rank=0,
        local_rank=0,
        group_name="tp-fallback",
    )


def tp_group_for_config(parallel_config: ParallelConfig) -> GroupCoordinator:
    """模型构造用 TP 组视图：按 ``ParallelConfig`` 解析，而非读全局隐式状态。

    - 全局 model-parallel 已 init 且 ``tp_size`` 与 config 匹配 → 复用真 TP 组
      （含 ``device_group``，forward 时集体通信可用）。
    - 否则 → 按 config 建无 PG 视图（``ranks=[0..tp_size)``、rank 取
      ``resolve_rank``）：仅用于 meta 权重分片形状与 ``tp_rank`` 偏移；
      真 TP>1 forward 的 all-reduce/all-gather 需先
      ``ensure_model_parallel_initialized(config)`` 建真组。

    让模型层并行模式由显式 ``ParallelConfig`` 驱动，对齐「config 注入优先于
    全局状态」的边界；与 vLLM/SGLang 全局 parallel_state 的差异：lake 把
    配置作为构造期权威，全局组只在真分布式 boot 时才接管。
    """
    if model_parallel_is_initialized():
        tp = get_tp_group()
        if tp.world_size != parallel_config.tensor_parallel_size:
            raise RuntimeError(
                f"global tp_size={tp.world_size} != "
                f"ParallelConfig.tensor_parallel_size="
                f"{parallel_config.tensor_parallel_size}"
            )
        return tp
    rank = parallel_config.resolve_rank() if parallel_config.tensor_parallel_size > 1 else 0
    return GroupCoordinator(
        ranks=list(range(parallel_config.tensor_parallel_size)),
        rank=rank,
        local_rank=parallel_config.resolve_local_rank(),
        group_name="tp-config",
    )


# ---------------------------------------------------------------------------
# Process-global state
# ---------------------------------------------------------------------------

_TP: Optional[GroupCoordinator] = None
_PP: Optional[GroupCoordinator] = None
_DP: Optional[GroupCoordinator] = None
_WORLD: Optional[GroupCoordinator] = None
_PARALLEL_CONFIG: Optional[ParallelConfig] = None
_DISTRIBUTED_INITIALIZED: bool = False


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def model_parallel_is_initialized() -> bool:
    return _TP is not None and _PP is not None and _DP is not None


def get_tensor_model_parallel_world_size() -> int:
    return get_tp_group().world_size


def get_tensor_model_parallel_rank() -> int:
    return get_tp_group().rank_in_group


def get_data_parallel_world_size() -> int:
    return get_dp_group().world_size


def get_data_parallel_rank() -> int:
    return get_dp_group().rank_in_group


def get_pipeline_model_parallel_world_size() -> int:
    return get_pp_group().world_size


def get_pipeline_model_parallel_rank() -> int:
    return get_pp_group().rank_in_group


def get_parallel_config() -> ParallelConfig:
    assert _PARALLEL_CONFIG is not None, "parallel config is not initialized"
    return _PARALLEL_CONFIG


def _maybe_new_group(ranks: Sequence[int], backend: str) -> object | None:
    """多 rank 且 dist 已 init 时建子组；否则返回 None（单测/单卡路径）。"""
    if len(ranks) <= 1:
        return None
    try:
        import torch.distributed as dist
    except ImportError:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.new_group(ranks=list(ranks), backend=backend)


def _make_coordinator(
    ranks: Sequence[int],
    *,
    global_rank: int,
    local_rank: int,
    group_name: str,
    backend: str,
    create_pg: bool,
) -> GroupCoordinator:
    device_group = None
    cpu_group = None
    if create_pg and len(ranks) > 1:
        device_group = _maybe_new_group(ranks, backend)
        # CPU 侧 gloo 组留给对象广播 / barrier；device 不可用时也先不建
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized() and dist.is_gloo_available():
                cpu_group = dist.new_group(ranks=list(ranks), backend="gloo")
        except Exception:
            cpu_group = None
    return GroupCoordinator(
        ranks=list(ranks),
        rank=global_rank,
        local_rank=local_rank,
        group_name=group_name,
        device_group=device_group,
        cpu_group=cpu_group,
    )


def init_distributed_environment(
    parallel_config: ParallelConfig,
    *,
    backend: str = "gloo",
    distributed_init_method: Optional[str] = None,
) -> None:
    """初始化 torch process group（仅 ``world_size_across_dp > 1``）。

    单进程默认 backend=gloo，便于 CPU CI；GPU 多卡由调用方传 ``nccl``。
    已初始化则校验 world/rank 一致后直接返回。
    """
    global _DISTRIBUTED_INITIALIZED, _PARALLEL_CONFIG

    _PARALLEL_CONFIG = parallel_config
    world = parallel_config.world_size_across_dp
    if world <= 1:
        _DISTRIBUTED_INITIALIZED = False
        return

    import torch.distributed as dist

    rank = parallel_config.resolve_rank()
    local_rank = parallel_config.resolve_local_rank()
    if distributed_init_method is None:
        distributed_init_method = (
            f"tcp://{parallel_config.master_addr}:{parallel_config.master_port}"
        )

    if dist.is_initialized():
        if dist.get_world_size() != world or dist.get_rank() != rank:
            raise RuntimeError(
                f"torch.distributed already initialized with "
                f"world={dist.get_world_size()} rank={dist.get_rank()}, "
                f"expected world={world} rank={rank}"
            )
        _DISTRIBUTED_INITIALIZED = True
        return

    if not dist.is_backend_available(backend):
        if not dist.is_gloo_available():
            raise RuntimeError(
                f"backend {backend!r} unavailable and gloo fallback missing"
            )
        backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method=distributed_init_method,
        world_size=world,
        rank=rank,
    )
    if parallel_config.local_rank < 0:
        parallel_config.local_rank = local_rank
    if parallel_config.rank < 0:
        parallel_config.rank = rank
    _DISTRIBUTED_INITIALIZED = True


def initialize_model_parallel(
    parallel_config: Optional[ParallelConfig] = None,
    *,
    backend: Optional[str] = None,
) -> None:
    """按 mesh 切出 TP / PP / DP ``GroupCoordinator``。

    - ``world_size_across_dp == 1``：平凡组，不要求 dist。
    - 否则：要求已 ``init_distributed_environment``（或外部已 init dist）。
    """
    global _TP, _PP, _DP, _WORLD, _PARALLEL_CONFIG

    if model_parallel_is_initialized():
        raise RuntimeError("model parallel groups are already initialized")

    cfg = parallel_config or _PARALLEL_CONFIG or ParallelConfig()
    _PARALLEL_CONFIG = cfg

    world = cfg.world_size_across_dp
    rank = cfg.resolve_rank() if world > 1 else 0
    local_rank = cfg.resolve_local_rank()

    if world > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "initialize_model_parallel requires init_distributed_environment "
                "when world_size_across_dp > 1"
            )
        rank = dist.get_rank()
        world = dist.get_world_size()
        expected = cfg.world_size_across_dp
        if world != expected:
            raise RuntimeError(
                f"dist world_size={world} != ParallelConfig.world_size_across_dp={expected}"
            )
        backend = backend or dist.get_backend()
    else:
        backend = backend or "gloo"
        rank = 0
        local_rank = 0

    create_pg = world > 1
    all_ranks = list(range(world))
    _WORLD = _make_coordinator(
        all_ranks,
        global_rank=rank,
        local_rank=local_rank,
        group_name="world",
        backend=backend,
        create_pg=False,  # WORLD 用默认 PG
    )
    if create_pg:
        try:
            import torch.distributed as dist

            _WORLD.device_group = dist.group.WORLD
        except Exception:
            pass

    tp_groups = tp_group_ranks(
        cfg.data_parallel_size, cfg.pipeline_parallel_size, cfg.tensor_parallel_size
    )
    pp_groups = pp_group_ranks(
        cfg.data_parallel_size, cfg.pipeline_parallel_size, cfg.tensor_parallel_size
    )
    dp_groups = dp_group_ranks(
        cfg.data_parallel_size, cfg.pipeline_parallel_size, cfg.tensor_parallel_size
    )

    _TP = _make_coordinator(
        find_group_containing(tp_groups, rank),
        global_rank=rank,
        local_rank=local_rank,
        group_name="tp",
        backend=backend,
        create_pg=create_pg,
    )
    _PP = _make_coordinator(
        find_group_containing(pp_groups, rank),
        global_rank=rank,
        local_rank=local_rank,
        group_name="pp",
        backend=backend,
        create_pg=create_pg,
    )
    _DP = _make_coordinator(
        find_group_containing(dp_groups, rank),
        global_rank=rank,
        local_rank=local_rank,
        group_name="dp",
        backend=backend,
        create_pg=create_pg,
    )


def ensure_model_parallel_initialized(
    parallel_config: ParallelConfig,
    *,
    backend: str = "gloo",
) -> None:
    """Boot 挂点：幂等初始化 dist（如需）+ model parallel groups。"""
    global _PARALLEL_CONFIG
    _PARALLEL_CONFIG = parallel_config
    if not model_parallel_is_initialized():
        if parallel_config.is_distributed and not _DISTRIBUTED_INITIALIZED:
            init_distributed_environment(parallel_config, backend=backend)
        initialize_model_parallel(parallel_config, backend=backend)


def destroy_model_parallel() -> None:
    """测试 / 进程退出时清全局组状态（不 destroy 外部 WORLD PG）。"""
    global _TP, _PP, _DP, _WORLD, _PARALLEL_CONFIG, _DISTRIBUTED_INITIALIZED
    _TP = None
    _PP = None
    _DP = None
    _WORLD = None
    _PARALLEL_CONFIG = None
    _DISTRIBUTED_INITIALIZED = False
