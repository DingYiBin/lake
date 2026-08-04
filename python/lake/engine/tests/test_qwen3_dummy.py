"""Qwen3 model backend skeleton."""

from __future__ import annotations

import os

import torch
from transformers import Qwen3Config

from lake.engine.agents.memory import InMemoryAgent
from lake.engine.model_runner import ModelRunner
from lake.engine.model_executor.models.loader import DummyModelLoader, get_model_loader

from lake.engine.model_executor.models.qwen.qwen3 import (
    Qwen3ForCausalLM,
)
from lake.engine.model_executor.models.registry import (
    ModelRegistry,
    load_hf_config,
)
from lake.engine.pool_iface import PoolIface
from lake.engine.pool_types import ReadyHandle
from lake.runtime.req import Req
from lake.runtime.node_scheduler import NodeScheduler, build_req_from_generate
from lake.runtime.role import RoleConfig
from lake.runtime.scheduler_output import ForwardMode, GrammarOutput, SamplingParams, SchedulerOutput


QWEN3_0_6B_MODEL_ID = os.path.expanduser(
    os.environ.get("LAKE_TEST_QWEN3_MODEL_PATH", "Qwen/Qwen3-0.6B")
)
QWEN3_0_6B_CONFIG = load_hf_config(QWEN3_0_6B_MODEL_ID)


def test_qwen3_config_matches_dense_0_6b_shape() -> None:
    cfg = QWEN3_0_6B_CONFIG
    assert isinstance(cfg, Qwen3Config)
    assert type(cfg).__module__.startswith("transformers.")
    assert cfg.model_type == "qwen3"
    assert cfg.architectures == ["Qwen3ForCausalLM"]
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.vocab_size == 151936
    assert cfg.max_position_embeddings == 40960
    dtype = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    assert dtype is torch.bfloat16


def test_dummy_model_loader_loads_qwen3() -> None:
    loader = DummyModelLoader()
    model = loader.load_model(Qwen3ForCausalLM, QWEN3_0_6B_CONFIG)
    assert model.loaded_dummy_weights is True
    assert "model.embed_tokens.weight" in model.loaded_weights
    assert "model.layers.0.self_attn.qkv_proj.weight" in model.loaded_weights
    assert "model.layers.0.mlp.gate_up_proj.weight" in model.loaded_weights


def test_default_model_loader_is_real_weight_boundary() -> None:
    loader = get_model_loader("hf", model_path=QWEN3_0_6B_MODEL_ID)
    try:
        loader.load_model(Qwen3ForCausalLM, QWEN3_0_6B_CONFIG)
        raise AssertionError("expected real weight loader to be pending")
    except NotImplementedError as e:
        assert "real weight loading" in str(e)


def test_qwen3_load_weights_keeps_model_api_plain() -> None:
    model = Qwen3ForCausalLM(QWEN3_0_6B_CONFIG)
    assert hasattr(model, "training")
    assert model.model.config is QWEN3_0_6B_CONFIG
    assert len(model.model.layers) == QWEN3_0_6B_CONFIG.num_hidden_layers
    assert model.model.layers[0].layer_idx == 0
    assert model.lm_head is model.model.embed_tokens
    # C16a：forward/compute_logits 现为真实计算（需物化权重）；未物化时 meta 权重不可 forward，
    # 故此处只验 API 边界，真实 forward 由 materialize 后的用例覆盖。
    assert callable(model.forward)
    assert callable(model.compute_logits)
    loaded = model.load_weights([("model.norm.weight", object())])
    assert model.loaded_dummy_weights is False
    assert loaded == {"model.norm.weight"}


def test_qwen3_module_tree_matches_vllm_packed_layout() -> None:
    from lake.engine.model_executor.layers.linear import (
        ColumnParallelLinearLayer,
        MergedColumnParallelLinearLayer,
        RowParallelLinearLayer,
    )

    model = Qwen3ForCausalLM(QWEN3_0_6B_CONFIG)
    layer0 = model.model.layers[0]
    head_dim = QWEN3_0_6B_CONFIG.head_dim
    q_size = QWEN3_0_6B_CONFIG.num_attention_heads * head_dim
    kv_size = QWEN3_0_6B_CONFIG.num_key_value_heads * head_dim

    assert model.model.embed_tokens.weight.shape == (
        QWEN3_0_6B_CONFIG.vocab_size,
        QWEN3_0_6B_CONFIG.hidden_size,
    )
    assert isinstance(layer0.self_attn.qkv_proj, ColumnParallelLinearLayer)
    assert isinstance(layer0.self_attn.o_proj, RowParallelLinearLayer)
    assert isinstance(layer0.mlp.gate_up_proj, MergedColumnParallelLinearLayer)
    assert layer0.mlp.gate_up_proj.output_sizes == [
        QWEN3_0_6B_CONFIG.intermediate_size,
        QWEN3_0_6B_CONFIG.intermediate_size,
    ]
    assert isinstance(layer0.mlp.down_proj, RowParallelLinearLayer)
    assert layer0.self_attn.qkv_proj.weight.shape == (
        q_size + (2 * kv_size),
        QWEN3_0_6B_CONFIG.hidden_size,
    )
    assert layer0.self_attn.o_proj.weight.shape == (
        QWEN3_0_6B_CONFIG.hidden_size,
        q_size,
    )
    assert layer0.self_attn.num_heads == QWEN3_0_6B_CONFIG.num_attention_heads
    assert layer0.self_attn.num_kv_heads == QWEN3_0_6B_CONFIG.num_key_value_heads
    assert layer0.self_attn.q_norm.weight.shape == (head_dim,)
    assert layer0.self_attn.k_norm.weight.shape == (head_dim,)
    assert layer0.mlp.gate_up_proj.weight.shape == (
        2 * QWEN3_0_6B_CONFIG.intermediate_size,
        QWEN3_0_6B_CONFIG.hidden_size,
    )
    assert layer0.mlp.down_proj.weight.shape == (
        QWEN3_0_6B_CONFIG.hidden_size,
        QWEN3_0_6B_CONFIG.intermediate_size,
    )
    assert layer0.input_layernorm.weight.shape == (QWEN3_0_6B_CONFIG.hidden_size,)
    assert layer0.post_attention_layernorm.weight.shape == (
        QWEN3_0_6B_CONFIG.hidden_size,
    )
    assert model.model.norm.weight.shape == (QWEN3_0_6B_CONFIG.hidden_size,)
    assert model.lm_head is model.model.embed_tokens


def test_qwen3_load_model_rejects_unsupported_architecture() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    unsupported = ModelRunner(pool, model_config=UnitUnsupportedConfig())
    try:
        unsupported.load_model(model_path="unit/unsupported")
        raise AssertionError("expected unsupported architecture")
    except NotImplementedError as e:
        assert "Model architectures" in str(e)


class UnitCustomForCausalLM:
    def __init__(self, config) -> None:
        self.config = config
        self.loaded_weights: set[str] = set()

    def state_dict(self) -> dict[str, object]:
        return {"unit.weight": object()}

    def load_weights(self, weights) -> set[str]:
        self.loaded_weights = {name for name, _ in weights}
        return set(self.loaded_weights)


class UnitCustomConfig:
    architectures = ["UnitCustomForCausalLM"]


class UnitUnsupportedConfig:
    architectures = ["UnitUnsupportedForCausalLM"]


def test_model_runner_uses_registered_architecture_for_load_model() -> None:
    old = ModelRegistry.models.get("UnitCustomForCausalLM")
    ModelRegistry.register_model("UnitCustomForCausalLM", UnitCustomForCausalLM)
    try:
        ag = InMemoryAgent()
        pool = PoolIface(ag)
        runner = ModelRunner(pool, model_config=UnitCustomConfig())
        info = runner.load_model(model_path="unit/custom", revision="r1")
        assert info.model_path == "unit/custom"
        assert info.served_model_name == "model"
        assert info.revision == "r1"
        assert info.architecture == "UnitCustomForCausalLM"
        assert info.load_format == "dummy"
        assert info.load_dummy_weights is True
    finally:
        if old is None:
            ModelRegistry.models.pop("UnitCustomForCausalLM", None)
        else:
            ModelRegistry.models["UnitCustomForCausalLM"] = old


def test_scheduler_qwen3_dummy_finishes() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    role = RoleConfig(
        model_path=QWEN3_0_6B_MODEL_ID,
        enable_overlap=False,
        max_running_reqs=2,
    )
    runner = ModelRunner(pool)
    runner.load_model(model_path=QWEN3_0_6B_MODEL_ID)
    sched = NodeScheduler(pool, runner, role)
    sched.add_request(build_req_from_generate("q1", "model", list(range(8)), 3, "n0"))
    sched.run_until_idle()
    done = sched.get_req("q1")
    assert done.finished
    assert done.num_output_tokens == 3
    assert all(0 <= t < QWEN3_0_6B_CONFIG.vocab_size for t in done.output_token_ids)
    assert ag.finished == ["q1"]


def test_qwen3_dummy_decode_uses_sampling_bitmask() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    runner = ModelRunner(pool)
    runner.load_model(model_path=QWEN3_0_6B_MODEL_ID)
    req = Req(
        req_id="q-mask",
        served_model_name="model",
        prompt_token_ids=[3],
        sampling_params=SamplingParams(max_new_tokens=1, structured_output="json"),
    )
    req.num_computed_tokens = 1
    forced_token = 7
    mask = [False] * QWEN3_0_6B_CONFIG.vocab_size
    mask[forced_token] = True
    output = SchedulerOutput(
        step_id=11,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={req.req_id: 1},
        total_num_scheduled_tokens=1,
        req_forward_modes={req.req_id: ForwardMode.DECODE},
        req_num_computed_at_schedule={req.req_id: 1},
        grammar_output=GrammarOutput(
            req_ids=[req.req_id],
            token_bitmask_by_req={req.req_id: mask},
        ),
        has_structured_output=True,
    )
    ready = ReadyHandle(
        step_id=11,
        block_table_by_req={req.req_id: [0]},
        slot_mapping_by_req={req.req_id: [0]},
    )

    out = runner.execute_model(output, ready, {req.req_id: req})
    assert out.next_token_ids == {req.req_id: [forced_token]}
    assert ag.done_calls == 0


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        architectures=["Qwen3ForCausalLM"],
    )


def test_qwen3_materialize_and_forward_produces_finite_logits() -> None:
    """C16a：物化 meta 权重后真实 forward 产有限 logits。"""
    from lake.engine.model_executor.layers.attentions import build_attn_backend
    from lake.engine.model_executor.models.loader import materialize_model

    model = Qwen3ForCausalLM(_tiny_qwen3_config(), attn_backend=build_attn_backend("cpu"))
    materialize_model(model, device="cpu", dtype=torch.float32)
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    positions = torch.arange(4, dtype=torch.long)
    with torch.no_grad():
        hidden = model.forward(input_ids, positions)
        logits = model.compute_logits(hidden)
    assert hidden.shape == (4, 32)
    assert logits.shape == (4, 64)
    assert torch.isfinite(logits).all()


def test_qwen3_forward_argmax_in_vocab_range() -> None:
    """C16a：真实 forward 末位 logits 的 argmax 落在 vocab 内且确定性。"""
    from lake.engine.model_executor.layers.attentions import build_attn_backend
    from lake.engine.model_executor.models.loader import materialize_model

    cfg = _tiny_qwen3_config()
    model = Qwen3ForCausalLM(cfg, attn_backend=build_attn_backend("cpu"))
    materialize_model(model, device="cpu", dtype=torch.float32, seed=1)
    input_ids = torch.tensor([5, 6, 7], dtype=torch.long)
    positions = torch.arange(3, dtype=torch.long)
    with torch.no_grad():
        logits = model.compute_logits(model.forward(input_ids, positions))
    tok = int(torch.argmax(logits[-1]).item())
    assert 0 <= tok < cfg.vocab_size
    # 同输入再跑一次，确定性
    with torch.no_grad():
        logits2 = model.compute_logits(model.forward(input_ids, positions))
    assert int(torch.argmax(logits2[-1]).item()) == tok


def test_runner_dummy_run_produces_real_forward_tokens() -> None:
    """C16a：runner dummy_run 经真实 forward 产 in-range token。"""
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    runner = ModelRunner(pool)
    runner.load_model(model_path=QWEN3_0_6B_MODEL_ID)
    out = runner.dummy_run(num_reqs=1, tokens_per_req=4, step_id=0)
    assert len(out.next_token_ids) == 1
    tok = next(iter(out.next_token_ids.values()))[0]
    assert 0 <= tok < QWEN3_0_6B_CONFIG.vocab_size
