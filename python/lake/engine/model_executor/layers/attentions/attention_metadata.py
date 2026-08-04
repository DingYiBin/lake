"""AttentionMetadata（D4 / C8 / device 化 P1）。

对照 vLLM ``AttentionMetadata``：runner 填 seq/query 几何；
**block table 由 agent 经 ReadyHandle 挂载**，引擎只读。
热路径字段为 device tensor（见 ``docs/architecture/input-batch-device.md``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from lake.engine.input_batch import InputBuffers


@dataclass
class AttentionMetadata:
    """本步 attention 输入（无物理地址权威）。"""

    # host 控制视图（按 req_id）
    seq_lens: Dict[str, int] = field(default_factory=dict)
    query_start: Dict[str, int] = field(default_factory=dict)
    query_end: Dict[str, int] = field(default_factory=dict)
    block_tables: Dict[str, List[int]] = field(default_factory=dict)

    # device / cpu tensor 热路径（与 req_order 对齐）
    query_start_loc: torch.Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.int32)
    )
    seq_lens_ordered: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int32)
    )
    slot_mapping: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int32)
    )
    positions: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int64)
    )
    # [num_reqs, max_blocks]；有效长度见 block_table_lens
    block_table_tensor: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, 0, dtype=torch.int32)
    )
    block_table_lens: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int32)
    )

    max_seq_len: int = 0
    max_query_len: int = 0
    num_reqs: int = 0
    num_actual_tokens: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    # P3.1 固定 shape padding：forward 吃 padded shape（num_reqs/num_actual_tokens
    # = padded），sampler/host 用 actual_*。padding 关时 padded=actual。
    actual_num_reqs: int = 0
    actual_num_tokens: int = 0
    padded_num_reqs: int = 0
    padded_num_tokens: int = 0
    is_padding: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool)
    )


def build_attn_metadata(
    *,
    seq_lens: Dict[str, int],
    query_start: Dict[str, int],
    query_end: Dict[str, int],
    block_tables: Optional[Dict[str, List[int]]] = None,
    buffers: Optional[InputBuffers] = None,
    slot_mapping_by_req: Optional[Dict[str, List[int]]] = None,
    req_order: Optional[List[str]] = None,
) -> AttentionMetadata:
    order = req_order or list(seq_lens.keys())
    tables = block_tables or {}
    max_seq = 0
    max_q = 0
    for rid in order:
        qs = query_start.get(rid, 0)
        qe = query_end.get(rid, qs)
        q_len = max(0, qe - qs)
        max_seq = max(max_seq, seq_lens.get(rid, 0))
        max_q = max(max_q, q_len)

    if buffers is not None:
        # materialize 已把 seq_lens 行写成 query_end；runner 传入的 seq_lens dict 与之对齐。
        n = buffers.effective_num_reqs
        nt = buffers.effective_num_tokens
        device = buffers.device
        return AttentionMetadata(
            seq_lens=dict(seq_lens),
            query_start=dict(query_start),
            query_end=dict(query_end),
            block_tables=dict(tables),
            query_start_loc=buffers.query_start_loc[: n + 1],
            seq_lens_ordered=buffers.seq_lens[:n],
            slot_mapping=buffers.slot_mapping[:nt],
            positions=buffers.positions[:nt],
            block_table_tensor=buffers.block_table[:n],
            block_table_lens=buffers.block_table_lens[:n],
            max_seq_len=max_seq,
            max_query_len=max_q,
            num_reqs=n,
            num_actual_tokens=nt,
            actual_num_reqs=buffers.num_reqs,
            actual_num_tokens=buffers.num_tokens,
            padded_num_reqs=n,
            padded_num_tokens=nt,
            is_padding=buffers.is_padding[:nt],
            device=device,
        )

    # 无 buffer：在 CPU 上现建（单测 / 轻路径）
    qsl_list = [0]
    flat_slots: List[int] = []
    positions: List[int] = []
    table_rows: List[List[int]] = []
    for rid in order:
        qs = query_start.get(rid, 0)
        qe = query_end.get(rid, qs)
        q_len = max(0, qe - qs)
        qsl_list.append(qsl_list[-1] + q_len)
        req_slots = (slot_mapping_by_req or {}).get(rid)
        if req_slots is not None and len(req_slots) != q_len:
            raise ValueError(
                f"slot_mapping len mismatch req={rid}: {len(req_slots)} != {q_len}"
            )
        flat_slots.extend(req_slots if req_slots is not None else range(qs, qe))
        positions.extend(range(qs, qe))
        table_rows.append(list(tables.get(rid, [])))

    max_blocks = max((len(t) for t in table_rows), default=0)
    bt = torch.zeros(len(order), max(max_blocks, 1), dtype=torch.int32)
    btl = torch.zeros(len(order), dtype=torch.int32)
    for i, row in enumerate(table_rows):
        if row:
            bt[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
        btl[i] = len(row)

    actual_n = len(order)
    actual_nt = qsl_list[-1] if qsl_list else 0
    return AttentionMetadata(
        seq_lens=dict(seq_lens),
        query_start=dict(query_start),
        query_end=dict(query_end),
        block_tables=dict(tables),
        query_start_loc=torch.tensor(qsl_list, dtype=torch.int32),
        seq_lens_ordered=torch.tensor(
            [seq_lens.get(rid, 0) for rid in order], dtype=torch.int32
        ),
        slot_mapping=torch.tensor(flat_slots, dtype=torch.int32),
        positions=torch.tensor(positions, dtype=torch.int64),
        block_table_tensor=bt if order else torch.zeros(0, 0, dtype=torch.int32),
        block_table_lens=btl,
        max_seq_len=max_seq,
        max_query_len=max_q,
        num_reqs=actual_n,
        num_actual_tokens=actual_nt,
        actual_num_reqs=actual_n,
        actual_num_tokens=actual_nt,
        padded_num_reqs=actual_n,
        padded_num_tokens=actual_nt,
        is_padding=torch.zeros(actual_nt, dtype=torch.bool),
        device=torch.device("cpu"),
    )
