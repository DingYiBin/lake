"""薄 ModelRunner：consume ready → prepare → forward → sample。

对齐 vLLM `GPUModelRunner.execute_model`：统一入口，按本步 token 几何执行
（`num_scheduled_tokens` / `req_num_computed_at_schedule`），不按 SGLang 分相状态机。
C8：`prepare_inputs` / `prepare_attn` / `sample_tokens` 拆步；残差 query 路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch

from lake.engine.model_executor.layers.attentions import (
    AttentionMetadata,
    build_attn_backend,
    build_attn_metadata,
)
from lake.engine.input_batch import InputBatch, InputBuffers
from lake.engine.model_executor.models.loader import materialize_model
from lake.engine.model_executor.models.registry import load_registered_model
from lake.engine.pool_iface import PoolIface
from lake.engine.pool_types import ReadyHandle
from lake.engine.sample.greedy import greedy_sample
from lake.engine.sample.grammar import apply_token_bitmask
from lake.runtime.req import Req
from lake.runtime.scheduler_output import ForwardMode, SamplingParams, SchedulerOutput


@dataclass
class ModelRunnerOutput:
    step_id: int
    next_token_ids: Dict[str, List[int]] = field(default_factory=dict)
    next_draft_tokens: Dict[str, List[int]] = field(default_factory=dict)
    architecture: str = ""


@dataclass(frozen=True)
class ModelLoadInfo:
    model_path: str
    served_model_name: str
    revision: str
    architecture: str
    load_format: str = "dummy"
    load_dummy_weights: bool = False
    weight_pinned: bool = False


@dataclass(frozen=True)
class ModelRunnerStatus:
    model_path: str
    served_model_name: str
    revision: str
    architecture: str
    loaded: bool
    warmed: bool


class ModelRunner:
    def __init__(
        self,
        pool: PoolIface,
        *,
        attn_backend_name: str = "cpu",
        model_config: Optional[Any] = None,
        weight_pin_callback: Optional[Callable[[ModelLoadInfo], None]] = None,
        pad_num_reqs: int = 0,
        pad_num_tokens: int = 0,
    ) -> None:
        self._pool = pool
        self._input_batch = InputBatch()
        self._input_buffers = InputBuffers(max_num_reqs=64, max_num_tokens=8192)
        # P3.1 固定 shape padding（graph capture 地基）：>0 时 forward 吃 padded shape。
        if pad_num_reqs > 0 or pad_num_tokens > 0:
            self._input_buffers.set_padding(pad_num_reqs, pad_num_tokens)
        self._attn_meta: Optional[AttentionMetadata] = None
        # attention 后端实例：runner 经注册表 ``build_attn_backend`` 建一次，沿模型树
        # 构造期注入到每个 ``Qwen3PagedAttention``——模型层不 import 任何具体后端
        # （对齐 vLLM ``Attention``/SGLang ``RadixAttention``：后端由 runner 选定注入）。
        self._attn_backend = build_attn_backend(attn_backend_name)
        self._config_override = model_config
        self._model: Optional[Any] = None
        self._architecture = ""
        self._weight_pin_callback = weight_pin_callback
        self._model_path = ""
        self._served_model_name = "model"
        self._model_revision = ""
        self._model_loaded = False
        self._model_warmed = False
        # C16b：mock KV arena（per-layer (k_cache, v_cache)）。生产由存储池放置；
        # 此处 runner 持 mock 句柄供 dummy/test paged 路径，arena 归属后置到池。
        self._kv_caches: Optional[List[Any]] = None

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def model_warmed(self) -> bool:
        return self._model_warmed

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def served_model_name(self) -> str:
        return self._served_model_name

    @property
    def architecture(self) -> str:
        return self._architecture

    def status(self) -> ModelRunnerStatus:
        return ModelRunnerStatus(
            model_path=self._model_path,
            served_model_name=self._served_model_name,
            revision=self._model_revision,
            architecture=self._architecture,
            loaded=self._model_loaded,
            warmed=self._model_warmed,
        )

    def load_model(
        self,
        *,
        model_path: str = "",
        served_model_name: str = "model",
        revision: str = "",
        load_format: str = "dummy",
        load_dummy_weights: bool = False,
        pin_weights: bool = True,
    ) -> ModelLoadInfo:
        """C12：真实模型加载骨架。

        对齐 vLLM `GPUModelRunner.load_model` 的阶段边界：先建立模型对象，
        再初始化依赖模型的执行组件。选模走 HF ``config.architectures`` → registry
        （未注册 raise）；``load_format=dummy`` 跳过权重下载（对齐 vLLM/SGLang
        ``DummyModelLoader``）。权重所有权仍归存储池。
        """

        loaded = load_registered_model(
            model_path=model_path,
            revision=revision,
            load_format=load_format,
            config_override=self._config_override,
            attn_backend=self._attn_backend,
        )
        self._model = loaded.model
        # C16a：dummy 路径物化 meta 权重到 cpu（小随机 init），使 forward 产真实 logits。
        # 真权重（load_format="hf"）由 DefaultModelLoader 装载，不物化。
        if loaded.load_format == "dummy":
            materialize_model(self._model, device="cpu")
        # C16b：分配 mock KV arena（per-layer k/v），供 paged forward_varlen 路径。
        self._kv_caches = self._allocate_kv_caches()
        self._architecture = loaded.architecture
        self._model_path = loaded.model_path
        self._served_model_name = served_model_name or "model"
        self._model_revision = loaded.revision
        self._model_loaded = True
        self._model_warmed = False
        info = ModelLoadInfo(
            model_path=self._model_path,
            served_model_name=self._served_model_name,
            revision=self._model_revision,
            architecture=self._architecture,
            load_format=loaded.load_format,
            load_dummy_weights=load_dummy_weights or loaded.load_dummy_weights,
            weight_pinned=pin_weights,
        )
        if pin_weights and self._weight_pin_callback is not None:
            self._weight_pin_callback(info)
        return info

    def _allocate_kv_caches(self) -> List[Any]:
        """C16b：分配 mock per-layer KV arena（``[total_slots, num_kv_heads, head_dim]``）。

        生产由存储池放置 HBM（方案 Z）；dummy/test 路径由 runner 持 mock 句柄。
        ``total_slots = max_num_blocks * block_size`` 覆盖 block_table 引用的全部 block。
        非 Qwen3 结构（自定义/测试桩无 ``model.layers``）返回空列表 → 走 C16a 回退。
        """
        assert self._model is not None
        model = getattr(self._model, "model", None)
        layers = getattr(model, "layers", None)
        if not layers:
            return []
        cfg = self._model.config
        num_layers = int(getattr(cfg, "num_hidden_layers", len(layers)))
        layer0 = layers[0]
        num_kv_heads = int(getattr(layer0.self_attn, "num_kv_heads", 0))
        head_dim = int(getattr(layer0.self_attn, "head_dim", 0))
        if num_kv_heads <= 0 or head_dim <= 0:
            return []
        dtype = next(self._model.parameters()).dtype
        device = next(self._model.parameters()).device
        total_slots = self._input_buffers.max_num_blocks * self._input_buffers.block_size
        return [
            (
                torch.zeros(total_slots, num_kv_heads, head_dim, dtype=dtype, device=device),
                torch.zeros(total_slots, num_kv_heads, head_dim, dtype=dtype, device=device),
            )
            for _ in range(num_layers)
        ]

    def warmup(
        self,
        *,
        num_reqs: int = 1,
        tokens_per_req: int = 1,
    ) -> ModelRunnerOutput:
        """C12：warmup 复用生产 dummy 入口，但跳过 pool.done。"""

        if not self._model_loaded:
            self.load_model(
                model_path=self._model_path,
                served_model_name=self._served_model_name,
            )
        out = self.dummy_run(
            num_reqs=num_reqs,
            tokens_per_req=tokens_per_req,
            step_id=-1,
        )
        self._model_warmed = True
        return out

    def prepare_inputs(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
    ) -> InputBatch:
        """对齐 vLLM `prepare_inputs`：组本步 InputBatch（无跨步 RequestState）。

        P2：行稠密追加（平行于 ``req_ids``），不再按 req_id 建 dict。
        """
        batch = InputBatch()
        spec_map = output.scheduled_spec_decode_tokens or {}
        for req_id, n in output.num_scheduled_tokens.items():
            if n <= 0:
                continue
            req = host_reqs.get(req_id)
            if req is None:
                continue
            prompt_len = len(req.prompt_token_ids)
            computed = output.req_num_computed_at_schedule.get(req_id, req.num_computed_tokens)
            query_start = output.req_query_start.get(req_id)
            query_end = output.req_query_end.get(req_id)
            if computed < prompt_len:
                start = computed if query_start is None else query_start
                end = min(prompt_len, query_end if query_end is not None else computed + n)
                tokens = list(req.prompt_token_ids[:end])
                qs = start
                qe = end
                is_prompt = True
            else:
                ctx = list(req.all_token_ids)
                draft = list(spec_map.get(req_id) or [])[: max(0, n - 1)]
                qs = query_start if query_start is not None else max(0, len(ctx) - 1)
                qe = query_end if query_end is not None else max(qs, qs + n)
                if draft:
                    # TARGET_VERIFY：输入 last_ctx + draft，产生 draft 校验 + bonus。
                    tokens = ctx + draft
                else:
                    tokens = ctx
                # overlap 下 scheduler 的 query 几何可能已包含 device-side inflight token；
                # Python 骨架尚无真实 device token 接力，用最后已知 token 占位以保持位置/slot 几何。
                if len(tokens) < qe:
                    pad = tokens[-1] if tokens else 0
                    tokens.extend([pad] * (qe - len(tokens)))
                tokens = tokens[:qe]
                is_prompt = False
            batch.add_request(
                req_id,
                num_scheduled_tokens=n,
                num_computed_tokens=computed,
                token_ids=tokens,
                query_start=qs,
                query_end=qe,
                is_prompt_phase=is_prompt,
            )
        self._input_batch = batch
        return batch

    def prepare_attn(self, batch: InputBatch, ready: ReadyHandle) -> AttentionMetadata:
        """对齐 vLLM `prepare_attn`：几何 + agent block table（只读）。

        ``build_attn_metadata`` 取 req_id-keyed 几何权威（亦可来自 scheduler），
        故此处由行稠密 batch 构一过性 dict 视图。
        """
        order = batch.req_ids
        seq_lens = {rid: batch.query_end[row] for row, rid in enumerate(order)}
        query_start = {rid: batch.query_start[row] for row, rid in enumerate(order)}
        query_end = {rid: batch.query_end[row] for row, rid in enumerate(order)}
        self._input_buffers.materialize(
            batch,
            slot_mapping_by_req=ready.slot_mapping_by_req,
            block_tables_by_req=ready.block_table_by_req,
        )
        meta = build_attn_metadata(
            seq_lens=seq_lens,
            query_start=query_start,
            query_end=query_end,
            block_tables=ready.block_table_by_req,
            buffers=self._input_buffers,
            req_order=order,
        )
        self._attn_meta = meta
        return meta

    def sample_tokens(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
        last_logits: Dict[str, List[float]],
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """从本步 logits 采样；接口预留与 `execute_model` 拆分（对齐 V2）。"""
        out: Dict[str, List[int]] = {}
        drafts_out: Dict[str, List[int]] = {}
        if output.scheduled_spec_decode_tokens:
            raise NotImplementedError("speculative verification requires a draft model")
        grammar = output.grammar_output
        deferred = set(grammar.deferred_req_ids if grammar is not None else [])
        bitmasks = grammar.token_bitmask_by_req if grammar is not None else {}
        for req_id, logits in last_logits.items():
            if req_id in deferred:
                continue
            req = host_reqs.get(req_id)
            if req is None:
                continue
            masked_logits = logits
            if req_id in bitmasks:
                masked_logits = apply_token_bitmask(logits, bitmasks[req_id])
            out[req_id] = [greedy_sample(masked_logits)]
        return out, drafts_out

    def execute_model(
        self,
        output: SchedulerOutput,
        ready: ReadyHandle,
        host_reqs: Optional[Mapping[str, Req]] = None,
    ) -> ModelRunnerOutput:
        if ready.step_id != output.step_id:
            raise RuntimeError(f"ready/output step mismatch: {ready.step_id} vs {output.step_id}")

        host = host_reqs or {}
        return self._execute_prepared(output, ready, host)

    def _execute_prepared(
        self,
        output: SchedulerOutput,
        ready: ReadyHandle,
        host: Mapping[str, Req],
    ) -> ModelRunnerOutput:
        """执行已 ready 的一步；不负责 pool.done。

        `execute_model` 和 `dummy_run` 共用本路径；真实 ready/done 生命周期
        由 NodeScheduler/RuntimeExecutor 边界统一收口，runner 不 ack pool。
        """

        if self._model is None:
            raise RuntimeError("model must be loaded before execute")

        batch = self.prepare_inputs(output, host)
        _meta = self.prepare_attn(batch, ready)
        next_tokens, next_drafts = self._forward_model(output, host, batch)

        return ModelRunnerOutput(
            step_id=output.step_id,
            next_token_ids=next_tokens,
            next_draft_tokens=next_drafts,
            architecture=self._architecture,
        )

    def _forward_model(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
        batch: InputBatch,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """C16b：batched paged forward（生产形态 dummy 路径）。

        本步所有 query token 拼成 flat ``input_ids``（来自 ``InputBuffers``，仅本步
        新算 token，非全上下文），经模型 forward：每层 attention 把新 k/v 按
        ``slot_mapping`` 写入 mock KV arena、``forward_varlen`` 读 paged 前缀 KV。
        取每请求末位 query token 的 logits 采样。非 prompt-phase 跳过采样（与 C16a
        一致：prefill 首 token 由后续 decode 产出）。无 KV arena 时回退到 C16a 逐请求
        非分页路径。
        """
        assert self._model is not None
        meta = self._attn_meta
        # paged 路径需 agent 出 block table；未出（测试桩 / 无 arena）回退 C16a 逐请求。
        if meta is None or not self._kv_caches or not meta.block_tables:
            return self._forward_model_per_req(output, host_reqs, batch)

        buffers = self._input_buffers
        nt = buffers.effective_num_tokens
        input_ids = buffers.input_ids[:nt].to(torch.long)
        positions = buffers.positions[:nt].to(torch.long)
        from lake.engine.model_executor.layers.attentions.context import forward_context
        with forward_context(meta, self._kv_caches):
            with torch.no_grad():
                hidden = self._model.forward(input_ids, positions)
                logits = self._model.compute_logits(hidden)  # [nt, vocab]

        qsl = meta.query_start_loc
        last_logits: Dict[str, List[float]] = {}
        for row, req_id in enumerate(batch.req_ids):
            if batch.is_prompt_phase[row]:
                continue
            last_idx = int(qsl[row + 1].item()) - 1
            if last_idx < 0:
                continue
            last_logits[req_id] = logits[last_idx].to(torch.float32).tolist()
        return self.sample_tokens(output, host_reqs, last_logits)

    def _forward_model_per_req(
        self,
        output: SchedulerOutput,
        host_reqs: Mapping[str, Req],
        batch: InputBatch,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """C16a 逐请求非分页 forward（无 KV arena 时的回退路径）。"""
        assert self._model is not None
        last_logits: Dict[str, List[float]] = {}
        for row, req_id in enumerate(batch.req_ids):
            if batch.is_prompt_phase[row]:
                continue
            req = host_reqs.get(req_id)
            if req is None:
                continue
            ctx = list(req.all_token_ids)
            if not ctx:
                continue
            input_ids = torch.tensor(ctx, dtype=torch.long)
            positions = torch.arange(len(ctx), dtype=torch.long)
            with torch.no_grad():
                hidden = self._model.forward(input_ids, positions)
                logits = self._model.compute_logits(hidden)  # [T, vocab]
            last_logits[req_id] = logits[-1].to(torch.float32).tolist()
        return self.sample_tokens(output, host_reqs, last_logits)

    def clear_drafter(self, req_id: str) -> None:
        return None

    def dummy_run(
        self,
        *,
        num_reqs: int = 1,
        tokens_per_req: int = 1,
        step_id: int = 0,
    ) -> ModelRunnerOutput:
        """对齐 vLLM `GPUModelRunner._dummy_run`：造假 SchedulerOutput 走生产入口。

        用于 warmup / graph capture 占位；构造 dummy host req / ready，不触发
        pool.prepare_step 或 pool.done。
        """
        if not self._model_loaded:
            raise ValueError("model must be loaded before dummy_run")
        num_reqs = max(1, int(num_reqs))
        tokens_per_req = max(1, int(tokens_per_req))
        num_tokens = {f"dummy-{i}": 1 for i in range(num_reqs)}
        host_reqs = {
            rid: Req(
                req_id=rid,
                served_model_name=self._served_model_name,
                prompt_token_ids=list(range(tokens_per_req)),
                sampling_params=SamplingParams(max_new_tokens=1),
            )
            for rid in num_tokens
        }
        output = SchedulerOutput(
            step_id=step_id,
            forward_mode=ForwardMode.DECODE,
            num_scheduled_tokens=num_tokens,
            total_num_scheduled_tokens=sum(num_tokens.values()),
            req_forward_modes={rid: ForwardMode.DECODE for rid in num_tokens},
            req_num_computed_at_schedule={rid: tokens_per_req for rid in num_tokens},
            req_query_start={rid: max(0, tokens_per_req - 1) for rid in num_tokens},
            req_query_end={rid: tokens_per_req for rid in num_tokens},
            can_run_graph=True,
        )
        ready = ReadyHandle(
            step_id=step_id,
            block_table_by_req={
                rid: list(range((tokens_per_req + 7) // 8)) for rid in num_tokens
            },
            slot_mapping_by_req={
                rid: [max(0, tokens_per_req - 1)] for rid in num_tokens
            },
        )
        return self._execute_prepared(output, ready, host_reqs)
