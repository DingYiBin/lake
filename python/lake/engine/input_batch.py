"""本步静态 / 批 buffer（非跨步请求权威）。

对齐 vLLM V2 ``InputBatch`` / ``InputBuffers`` 子集：
- ``InputBatch``：host 轻量几何（行稠密 list，平行于 ``req_ids``），供 materialize
- ``InputBuffers``：预分配 **device tensor**（固定地址）；CUDA 经 pin staging H2D

Host ``Req`` 权威仍在 ``node_scheduler``。计划见
``docs/architecture/input-batch-device.md``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch

DeviceLike = Union[str, torch.device]


@dataclass
class InputBatch:
    """本步 host 批几何（行稠密，对齐 vLLM V2 ``InputBatch`` 行索引形态）。

    所有按请求字段均为 ``list``、与 ``req_ids`` 行对齐（row = 在 ``req_ids``
    中的位置）。无跨步 ``RequestState``：本步构建、本步消费后丢弃。
    P2：由 dict-by-req_id 收敛为行稠密（见 ``input-batch-device.md``）。
    """

    req_ids: List[str] = field(default_factory=list)
    # [num_reqs] 行稠密，平行于 req_ids
    num_scheduled_tokens: List[int] = field(default_factory=list)
    num_computed_tokens: List[int] = field(default_factory=list)
    # 每请求：本步前向可见的 token 前缀（长度 = query_end）
    token_ids: List[List[int]] = field(default_factory=list)
    query_start: List[int] = field(default_factory=list)
    query_end: List[int] = field(default_factory=list)
    # prompt 相 vs 生成相（供 sample 跳过 extend）
    is_prompt_phase: List[bool] = field(default_factory=list)

    def clear(self) -> None:
        self.req_ids.clear()
        self.num_scheduled_tokens.clear()
        self.num_computed_tokens.clear()
        self.token_ids.clear()
        self.query_start.clear()
        self.query_end.clear()
        self.is_prompt_phase.clear()

    def add_request(
        self,
        req_id: str,
        *,
        num_scheduled_tokens: int,
        num_computed_tokens: int,
        token_ids: List[int],
        query_start: int,
        query_end: int,
        is_prompt_phase: bool,
    ) -> None:
        """按行追加一个请求的本步几何；所有字段平行于 ``req_ids``。"""
        self.req_ids.append(req_id)
        self.num_scheduled_tokens.append(num_scheduled_tokens)
        self.num_computed_tokens.append(num_computed_tokens)
        self.token_ids.append(token_ids)
        self.query_start.append(query_start)
        self.query_end.append(query_end)
        self.is_prompt_phase.append(is_prompt_phase)

    def index_of(self, req_id: str) -> int:
        """req_id -> 行下标（行稠密下按需查找；batch 量级小，O(n) 可接受）。"""
        return self.req_ids.index(req_id)


class InputBuffers:
    """固定地址执行 buffer（对齐 vLLM V2 ``InputBuffers``）。

    热路径字段为 device tensor；``device=cpu`` 时直接写入，便于单测。
    ``device=cuda`` 时先写 pin staging，再 ``non_blocking`` H2D。
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        *,
        device: DeviceLike = "cpu",
        max_num_blocks: int = 256,
    ) -> None:
        if max_num_reqs <= 0:
            raise ValueError("max_num_reqs must be > 0")
        if max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be > 0")
        if max_num_blocks <= 0:
            raise ValueError("max_num_blocks must be > 0")

        self.max_num_reqs = max_num_reqs
        self.max_num_tokens = max_num_tokens
        self.max_num_blocks = max_num_blocks
        self.device = torch.device(device)
        self._use_staging = self.device.type == "cuda"

        def _dev(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=self.device)

        self.input_ids = _dev((max_num_tokens,), torch.int32)
        self.positions = _dev((max_num_tokens,), torch.int64)
        self.is_padding = torch.ones(max_num_tokens, dtype=torch.bool, device=self.device)
        self.query_start_loc = _dev((max_num_reqs + 1,), torch.int32)
        self.seq_lens = _dev((max_num_reqs,), torch.int32)
        self.slot_mapping = torch.full(
            (max_num_tokens,), -1, dtype=torch.int32, device=self.device
        )
        self.block_table = _dev((max_num_reqs, max_num_blocks), torch.int32)
        self.block_table_lens = _dev((max_num_reqs,), torch.int32)

        self._stage: Optional[dict[str, torch.Tensor]] = None
        if self._use_staging:
            pin = dict(device="cpu", pin_memory=True)
            self._stage = {
                "input_ids": torch.zeros(max_num_tokens, dtype=torch.int32, **pin),
                "positions": torch.zeros(max_num_tokens, dtype=torch.int64, **pin),
                "is_padding": torch.ones(max_num_tokens, dtype=torch.bool, **pin),
                "query_start_loc": torch.zeros(max_num_reqs + 1, dtype=torch.int32, **pin),
                "seq_lens": torch.zeros(max_num_reqs, dtype=torch.int32, **pin),
                "slot_mapping": torch.full((max_num_tokens,), -1, dtype=torch.int32, **pin),
                "block_table": torch.zeros(
                    max_num_reqs, max_num_blocks, dtype=torch.int32, **pin
                ),
                "block_table_lens": torch.zeros(max_num_reqs, dtype=torch.int32, **pin),
            }

        self.req_ids: List[str] = []
        self.num_reqs: int = 0
        self.num_tokens: int = 0
        # P3.1 固定 shape padding：0=变长（effective=actual）；>0=pad 到固定值。
        # materialize 已把 pad 区置好（is_padding=True / slot_mapping=-1 / qsl 非递减 /
        # pad 行 seq_lens·block_table_lens=0）；此处只决定暴露给 forward 的 shape。
        self._padded_num_reqs: int = 0
        self._padded_num_tokens: int = 0

    def set_padding(self, padded_num_reqs: int, padded_num_tokens: int) -> None:
        """配置固定 shape padding（graph capture 地基）。

        0/0=关（变长）；>0 时 forward 永远看到 padded shape，is_padding 区分真实/pad。
        校验 ≤ max；materialize 时再校验 actual ≤ padded。
        """
        if (padded_num_reqs > 0) != (padded_num_tokens > 0):
            raise ValueError(
                "padded_num_reqs and padded_num_tokens must be both set or both 0"
            )
        if padded_num_reqs < 0 or padded_num_tokens < 0:
            raise ValueError(
                f"padded must be >= 0, got {padded_num_reqs}/{padded_num_tokens}"
            )
        if padded_num_reqs > self.max_num_reqs:
            raise ValueError(
                f"padded_num_reqs={padded_num_reqs} exceeds max_num_reqs={self.max_num_reqs}"
            )
        if padded_num_tokens > self.max_num_tokens:
            raise ValueError(
                f"padded_num_tokens={padded_num_tokens} exceeds max_num_tokens={self.max_num_tokens}"
            )
        self._padded_num_reqs = padded_num_reqs
        self._padded_num_tokens = padded_num_tokens

    @property
    def padding_enabled(self) -> bool:
        return self._padded_num_reqs > 0

    @property
    def effective_num_reqs(self) -> int:
        return self._padded_num_reqs if self._padded_num_reqs > 0 else self.num_reqs

    @property
    def effective_num_tokens(self) -> int:
        return self._padded_num_tokens if self._padded_num_tokens > 0 else self.num_tokens

    def _write_target(self, name: str) -> torch.Tensor:
        if self._stage is not None:
            return self._stage[name]
        return getattr(self, name)

    def clear(self) -> None:
        self.req_ids = []
        self.num_reqs = 0
        self.num_tokens = 0
        tgt_qsl = self._write_target("query_start_loc")
        tgt_seq = self._write_target("seq_lens")
        tgt_ids = self._write_target("input_ids")
        tgt_pos = self._write_target("positions")
        tgt_pad = self._write_target("is_padding")
        tgt_slot = self._write_target("slot_mapping")
        tgt_bt = self._write_target("block_table")
        tgt_btl = self._write_target("block_table_lens")
        tgt_qsl.zero_()
        tgt_seq.zero_()
        tgt_ids.zero_()
        tgt_pos.zero_()
        tgt_pad.fill_(True)
        tgt_slot.fill_(-1)
        tgt_bt.zero_()
        tgt_btl.zero_()

    def _flush_staging(self) -> None:
        if self._stage is None:
            return
        for name, staged in self._stage.items():
            getattr(self, name).copy_(staged, non_blocking=True)

    def materialize(
        self,
        batch: InputBatch,
        *,
        slot_mapping_by_req: Dict[str, List[int]] | None = None,
        block_tables_by_req: Dict[str, List[int]] | None = None,
    ) -> "InputBuffers":
        """把 ``InputBatch``（+ agent 表）写入静态 buffer。"""

        if len(batch.req_ids) > self.max_num_reqs:
            raise ValueError(
                f"num_reqs={len(batch.req_ids)} exceeds max_num_reqs={self.max_num_reqs}"
            )

        self.clear()
        self.req_ids = list(batch.req_ids)
        self.num_reqs = len(batch.req_ids)
        cursor = 0
        if self.padding_enabled:
            if self.num_reqs > self._padded_num_reqs:
                raise ValueError(
                    f"num_reqs={self.num_reqs} exceeds padded_num_reqs="
                    f"{self._padded_num_reqs}"
                )
        slots = slot_mapping_by_req or {}
        tables = block_tables_by_req or {}

        qsl = self._write_target("query_start_loc")
        seq = self._write_target("seq_lens")
        ids = self._write_target("input_ids")
        pos = self._write_target("positions")
        pad = self._write_target("is_padding")
        smap = self._write_target("slot_mapping")
        bt = self._write_target("block_table")
        btl = self._write_target("block_table_lens")

        for row, req_id in enumerate(batch.req_ids):
            qs = batch.query_start[row]
            qe = batch.query_end[row]
            q_len = max(0, qe - qs)
            if cursor + q_len > self.max_num_tokens:
                raise ValueError(
                    f"num_tokens={cursor + q_len} exceeds max_num_tokens={self.max_num_tokens}"
                )
            qsl[row] = cursor
            seq[row] = qe
            tokens = batch.token_ids[row][qs:qe]
            req_slots = slots.get(req_id)
            if req_slots is not None and len(req_slots) != q_len:
                raise ValueError(
                    f"slot_mapping len mismatch req={req_id}: {len(req_slots)} != {q_len}"
                )
            for j, token in enumerate(tokens):
                idx = cursor + j
                ids[idx] = int(token)
                pos[idx] = qs + j
                pad[idx] = False
                smap[idx] = int(req_slots[j]) if req_slots is not None else qs + j

            table = tables.get(req_id) or []
            if len(table) > self.max_num_blocks:
                raise ValueError(
                    f"block_table len={len(table)} exceeds max_num_blocks={self.max_num_blocks}"
                )
            if table:
                bt[row, : len(table)] = torch.tensor(table, dtype=torch.int32)
            btl[row] = len(table)
            cursor += q_len

        qsl[self.num_reqs] = cursor
        self.num_tokens = cursor
        for row in range(self.num_reqs + 1, self.max_num_reqs + 1):
            qsl[row] = cursor

        if self.padding_enabled and self.num_tokens > self._padded_num_tokens:
            raise ValueError(
                f"num_tokens={self.num_tokens} exceeds padded_num_tokens="
                f"{self._padded_num_tokens}"
            )

        self._flush_staging()
        return self
