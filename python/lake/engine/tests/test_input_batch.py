"""C8：InputBatch + AttentionMetadata + 同批两请求。"""

from __future__ import annotations

from lake.engine.agents.memory import InMemoryAgent
from lake.engine.model_executor.layers.attentions import build_attn_metadata
from lake.engine.input_batch import InputBatch, InputBuffers
from lake.engine.pool_iface import PoolIface
from lake.engine.pool_types import PreparePlan, ReadyHandle
from lake.kernels.attn_ref import causal_attn_queries
from lake.runtime.node_scheduler import NodeScheduler, build_req_from_generate
from lake.runtime.role import RoleConfig
from lake.runtime.scheduler_output import ForwardMode, ReqIoSet, SchedulerOutput
from lake.testing import make_runner


def test_causal_attn_queries_matches_full_slice() -> None:
    q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    k = list(q)
    v = list(q)
    full = __import__("lake.kernels.attn_ref", fromlist=["causal_attn"]).causal_attn(q, k, v)
    partial = causal_attn_queries(q[1:], k, v, q_pos_start=1)
    assert len(partial) == 2
    for a, b in zip(partial, full[1:]):
        assert all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def test_prepare_inputs_attn_metadata() -> None:
    pool = PoolIface(InMemoryAgent())
    runner = make_runner(pool, load=False)
    req = build_req_from_generate("r", "m", list(range(8)), 2, "n0")
    out = SchedulerOutput(
        step_id=1,
        forward_mode=ForwardMode.EXTEND,
        num_scheduled_tokens={"r": 3},
        total_num_scheduled_tokens=3,
        req_num_computed_at_schedule={"r": 2},
        req_forward_modes={"r": ForwardMode.EXTEND},
    )
    batch = runner.prepare_inputs(out, {"r": req})
    r = batch.index_of("r")
    assert batch.query_start[r] == 2 and batch.query_end[r] == 5
    assert batch.is_prompt_phase[r] is True
    ready = ReadyHandle(step_id=1, block_table_by_req={"r": [0, 1]})
    meta = runner.prepare_attn(batch, ready)
    assert meta.block_tables["r"] == [0, 1]
    assert meta.max_query_len == 3
    assert meta.query_start_loc.tolist() == [0, 3]
    assert meta.positions.tolist() == [2, 3, 4]
    assert meta.slot_mapping.tolist() == [2, 3, 4]
    assert meta.block_table_lens.tolist() == [2]
    assert meta.block_table_tensor[0, :2].tolist() == [0, 1]


def test_input_buffers_allocate_draft_state() -> None:
    buffers = InputBuffers(
        max_num_reqs=3,
        max_num_tokens=8,
        max_num_draft_tokens=2,
        max_num_mtp_layers=3,
        vocab_size=5,
        hidden_size=4,
    )

    assert buffers.draft.enabled is True
    assert buffers.draft.max_num_mtp_layers == 3
    assert tuple(buffers.draft.draft_token_ids.shape) == (3, 2)
    assert tuple(buffers.draft.draft_probs.shape) == (3, 2, 5)
    assert tuple(buffers.draft.hidden_states.shape) == (3, 3, 4)

    buffers.draft.num_draft_tokens[0] = 2
    buffers.draft.draft_token_ids[0, 0] = 7
    buffers.draft.bonus_token_ids[0] = 9
    buffers.draft.draft_probs[0, 0, 1] = 1.0
    buffers.draft.hidden_states[0, 0, 2] = 1.0

    buffers.clear()

    assert buffers.draft.num_draft_tokens.tolist() == [0, 0, 0]
    assert buffers.draft.draft_token_ids.tolist() == [[-1, -1], [-1, -1], [-1, -1]]
    assert buffers.draft.bonus_token_ids.tolist() == [-1, -1, -1]
    assert buffers.draft.draft_probs.sum().item() == 0.0
    assert buffers.draft.hidden_states.sum().item() == 0.0


def test_decode_slot_mapping_matches_query_position() -> None:
    pool = PoolIface(InMemoryAgent())
    runner = make_runner(pool, load=False)
    req = build_req_from_generate("r", "m", list(range(4)), 1, "n0")
    req.num_computed_tokens = len(req.prompt_token_ids)
    out = SchedulerOutput(
        step_id=2,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={"r": 1},
        total_num_scheduled_tokens=1,
        req_num_computed_at_schedule={"r": len(req.prompt_token_ids)},
        req_forward_modes={"r": ForwardMode.DECODE},
    )
    batch = runner.prepare_inputs(out, {"r": req})
    r = batch.index_of("r")
    assert batch.query_start[r] == 3
    assert batch.query_end[r] == 4

    ready = ReadyHandle(
        step_id=2,
        block_table_by_req={"r": [0]},
        slot_mapping_by_req={"r": [3]},
    )
    meta = runner.prepare_attn(batch, ready)
    assert meta.positions.tolist() == [3]
    assert meta.slot_mapping.tolist() == [3]


def test_prepare_inputs_uses_scheduler_query_geometry_under_overlap() -> None:
    pool = PoolIface(InMemoryAgent())
    runner = make_runner(pool, load=False)
    req = build_req_from_generate("r", "m", list(range(4)), 2, "n0")
    req.num_computed_tokens = len(req.prompt_token_ids)
    out = SchedulerOutput(
        step_id=3,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={"r": 1},
        total_num_scheduled_tokens=1,
        req_num_computed_at_schedule={"r": len(req.prompt_token_ids)},
        req_query_start={"r": 4},
        req_query_end={"r": 5},
        req_forward_modes={"r": ForwardMode.DECODE},
    )

    batch = runner.prepare_inputs(out, {"r": req})
    r = batch.index_of("r")
    assert batch.query_start[r] == 4
    assert batch.query_end[r] == 5
    assert len(batch.token_ids[r]) == 5

    ready = ReadyHandle(
        step_id=3,
        block_table_by_req={"r": [0]},
        slot_mapping_by_req={"r": [4]},
    )
    meta = runner.prepare_attn(batch, ready)
    assert meta.positions.tolist() == [4]
    assert meta.slot_mapping.tolist() == [4]


def test_two_reqs_same_batch() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    role = RoleConfig(
        enable_overlap=False,
        max_running_reqs=4,
        max_num_scheduled_tokens=64,
    )
    runner = make_runner(pool)
    sched = NodeScheduler(pool, runner, role)
    sched.add_request(build_req_from_generate("a", "m", list(range(6)), 2, "n0"))
    sched.add_request(build_req_from_generate("b", "m", list(range(6, 12)), 2, "n0"))
    out = sched.schedule()
    assert len(out.num_scheduled_tokens) == 2
    assert out.forward_mode == ForwardMode.EXTEND
    sched._run_batch(out)  # noqa: SLF001
    sched._pop_and_process()  # noqa: SLF001
    sched.run_until_idle()
    assert sched.get_req("a").finished and sched.get_req("b").finished
    assert set(ag.finished) == {"a", "b"}


def test_build_attn_metadata_loc() -> None:
    meta = build_attn_metadata(
        seq_lens={"a": 5, "b": 3},
        query_start={"a": 2, "b": 2},
        query_end={"a": 5, "b": 3},
        req_order=["a", "b"],
    )
    assert meta.query_start_loc.tolist() == [0, 3, 4]
    assert meta.max_query_len == 3


def test_input_buffers_materialize_ragged_queries() -> None:
    batch = InputBatch(
        req_ids=["a", "b"],
        token_ids=[[10, 11, 12, 13], [20, 21, 22]],
        query_start=[1, 2],
        query_end=[4, 3],
    )
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=8)
    buffers.materialize(
        batch,
        slot_mapping_by_req={"a": [101, 102, 103], "b": [201]},
        block_tables_by_req={"a": [0, 1], "b": [2]},
    )
    assert buffers.num_reqs == 2
    assert buffers.num_tokens == 4
    assert buffers.query_start_loc[:3].tolist() == [0, 3, 4]
    assert buffers.input_ids[:4].tolist() == [11, 12, 13, 22]
    assert buffers.positions[:4].tolist() == [1, 2, 3, 2]
    assert buffers.slot_mapping[:4].tolist() == [101, 102, 103, 201]
    assert buffers.is_padding[:4].tolist() == [False, False, False, False]
    assert buffers.block_table_lens[:2].tolist() == [2, 1]
    assert buffers.block_table[0, :2].tolist() == [0, 1]
    assert buffers.block_table[1, :1].tolist() == [2]


def test_build_attn_metadata_rejects_bad_slot_mapping() -> None:
    try:
        build_attn_metadata(
            seq_lens={"a": 4},
            query_start={"a": 1},
            query_end={"a": 4},
            slot_mapping_by_req={"a": [7]},
            req_order=["a"],
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "slot_mapping len mismatch" in str(e)


def test_inmemory_agent_returns_c11_tables() -> None:
    ag = InMemoryAgent()
    plan = PreparePlan(
        step_id=1,
        forward_mode=ForwardMode.EXTEND,
        read_set=[],
        write_set=[ReqIoSet(req_id="r1", token_start=0, token_end=9)],
        num_scheduled_tokens={"r1": 9},
    )
    ready = ag.prepare_step(plan)
    assert ready.block_table_by_req["r1"] == [0, 1]
    assert ready.slot_mapping_by_req["r1"] == list(range(9))


def test_input_buffers_padding_exposes_fixed_shape() -> None:
    batch = InputBatch(
        req_ids=["a", "b"],
        token_ids=[[10, 11, 12, 13], [20, 21, 22]],
        query_start=[1, 2],
        query_end=[4, 3],
    )
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=16)
    buffers.set_padding(padded_num_reqs=4, padded_num_tokens=8)
    buffers.materialize(
        batch,
        slot_mapping_by_req={"a": [101, 102, 103], "b": [201]},
        block_tables_by_req={"a": [0, 1], "b": [2]},
    )
    # actual 计数
    assert buffers.num_reqs == 2
    assert buffers.num_tokens == 4
    # effective（padded）shape
    assert buffers.effective_num_reqs == 4
    assert buffers.effective_num_tokens == 8
    assert buffers.padding_enabled is True
    # is_padding mask：真实 4 个 False，pad 4 个 True
    assert buffers.is_padding[:4].tolist() == [False, False, False, False]
    assert buffers.is_padding[4:8].tolist() == [True, True, True, True]
    # query_start_loc 非递减（pad req 零长 query，尾段填 num_tokens=4）
    qsl = buffers.query_start_loc[:5].tolist()
    assert qsl == [0, 3, 4, 4, 4]
    assert all(qsl[i] <= qsl[i + 1] for i in range(len(qsl) - 1))
    # pad 行 seq_lens / block_table_lens = 0
    assert buffers.seq_lens[:4].tolist() == [4, 3, 0, 0]
    assert buffers.block_table_lens[:4].tolist() == [2, 1, 0, 0]
    # pad 段 slot_mapping = -1
    assert buffers.slot_mapping[4:8].tolist() == [-1, -1, -1, -1]


def test_build_attn_metadata_padded_shape() -> None:
    batch = InputBatch(
        req_ids=["a", "b"],
        token_ids=[[10, 11, 12, 13], [20, 21, 22]],
        query_start=[1, 2],
        query_end=[4, 3],
    )
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=16)
    buffers.set_padding(padded_num_reqs=4, padded_num_tokens=8)
    buffers.materialize(
        batch,
        slot_mapping_by_req={"a": [101, 102, 103], "b": [201]},
        block_tables_by_req={"a": [0, 1], "b": [2]},
    )
    meta = build_attn_metadata(
        seq_lens={"a": 4, "b": 3},
        query_start={"a": 1, "b": 2},
        query_end={"a": 4, "b": 3},
        block_tables={"a": [0, 1], "b": [2]},
        buffers=buffers,
        req_order=["a", "b"],
    )
    # forward shape = padded
    assert meta.num_reqs == 4
    assert meta.num_actual_tokens == 8
    assert meta.padded_num_reqs == 4
    assert meta.padded_num_tokens == 8
    # sampler/host 真实计数
    assert meta.actual_num_reqs == 2
    assert meta.actual_num_tokens == 4
    # is_padding mask
    assert meta.is_padding.tolist() == [False, False, False, False, True, True, True, True]
    # tensor 切到 padded
    assert meta.query_start_loc.numel() == 5
    assert meta.seq_lens_ordered.numel() == 4
    assert meta.positions.numel() == 8


def test_padding_rejects_actual_exceeding_padded() -> None:
    batch = InputBatch(
        req_ids=["a", "b", "c"],
        token_ids=[[0], [1], [2]],
        query_start=[0, 0, 0],
        query_end=[1, 1, 1],
    )
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=16)
    buffers.set_padding(padded_num_reqs=2, padded_num_tokens=8)
    try:
        buffers.materialize(batch)
        raise AssertionError("expected ValueError for reqs > padded")
    except ValueError as e:
        assert "exceeds padded_num_reqs" in str(e)


def test_padding_rejects_tokens_exceeding_padded() -> None:
    batch = InputBatch(
        req_ids=["a"],
        token_ids=[list(range(10))],
        query_start=[0],
        query_end=[10],
    )
    buffers = InputBuffers(max_num_reqs=8, max_num_tokens=16)
    buffers.set_padding(padded_num_reqs=4, padded_num_tokens=8)
    try:
        buffers.materialize(batch)
        raise AssertionError("expected ValueError for tokens > padded")
    except ValueError as e:
        assert "exceeds padded_num_tokens" in str(e)


def test_set_padding_validates_bounds() -> None:
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=8)
    try:
        buffers.set_padding(padded_num_reqs=8, padded_num_tokens=4)
        raise AssertionError("expected ValueError for reqs > max")
    except ValueError as e:
        assert "exceeds max_num_reqs" in str(e)
    try:
        buffers.set_padding(padded_num_reqs=4, padded_num_tokens=16)
        raise AssertionError("expected ValueError for tokens > max")
    except ValueError as e:
        assert "exceeds max_num_tokens" in str(e)
    try:
        buffers.set_padding(padded_num_reqs=2, padded_num_tokens=0)
        raise AssertionError("expected ValueError for both-or-zero")
    except ValueError as e:
        assert "both set or both 0" in str(e)


def test_roleconfig_padding_validation() -> None:
    from lake.engine.config.role import RoleConfig

    RoleConfig(pad_num_reqs=0, pad_num_tokens=0)  # ok: off
    RoleConfig(pad_num_reqs=4, pad_num_tokens=8)  # ok: both set
    try:
        RoleConfig(pad_num_reqs=4, pad_num_tokens=0)
        raise AssertionError("expected ValueError for both-or-zero")
    except ValueError as e:
        assert "both set or both 0" in str(e)
    try:
        RoleConfig(pad_num_reqs=-1, pad_num_tokens=0)
        raise AssertionError("expected ValueError for negative")
    except ValueError as e:
        assert ">= 0" in str(e)


def test_roleconfig_padding_from_env() -> None:
    import os
    from lake.engine.config.role import RoleConfig

    old_r = os.environ.pop("LAKE_PAD_NUM_REQS", None)
    old_t = os.environ.pop("LAKE_PAD_NUM_TOKENS", None)
    try:
        os.environ["LAKE_PAD_NUM_REQS"] = "4"
        os.environ["LAKE_PAD_NUM_TOKENS"] = "8"
        role = RoleConfig.from_env()
        assert role.pad_num_reqs == 4
        assert role.pad_num_tokens == 8
    finally:
        os.environ.pop("LAKE_PAD_NUM_REQS", None)
        os.environ.pop("LAKE_PAD_NUM_TOKENS", None)
        if old_r is not None:
            os.environ["LAKE_PAD_NUM_REQS"] = old_r
        if old_t is not None:
            os.environ["LAKE_PAD_NUM_TOKENS"] = old_t
