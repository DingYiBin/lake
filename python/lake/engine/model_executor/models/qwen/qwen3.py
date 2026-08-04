"""Qwen3 model runtime skeleton.

类名与 vLLM `Qwen3ForCausalLM` 对齐；config 由模型加载侧从 Hugging Face
config 读取后传入。Linear 对齐 vLLM：``qkv_proj``/``gate_up_proj`` 用
ColumnParallel（QKV/MergedColumn 专用类后续替换）、``o_proj``/``down_proj``
用 RowParallel；``lm_head`` 未 tie 时用 ColumnParallel（占位 ParallelLMHead）。

本阶段经通用 `DummyModelLoader` 建立权重加载边界和 deterministic forward
占位，不加载真实 safetensors。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import Qwen3Config

from lake.engine.distributed.parallel_state import resolve_comm_group
from lake.engine.model_executor.layers.linear import (
    ColumnParallelLinearLayer,
    RowParallelLinearLayer,
)

if TYPE_CHECKING:
    from lake.engine.model_executor.layers.attentions import (
        AttentionBackend,
        AttentionMetadata,
    )


def _param_dtype(config: Qwen3Config) -> torch.dtype:
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str) and hasattr(torch, dtype):
        value = getattr(torch, dtype)
        if isinstance(value, torch.dtype):
            return value
    return torch.bfloat16


def _head_dim(config: Qwen3Config) -> int:
    return int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))


def _tp_world_size() -> int:
    """当前默认 TP 组大小（未 init parallel 时为 1）。"""
    return resolve_comm_group(None).world_size


class Qwen3RMSNorm(nn.Module):
    """RMSNorm parameter shell matching Qwen3/vLLM naming."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, device="meta", dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: object, residual: object | None = None) -> object:
        return hidden_states if residual is None else (hidden_states, residual)


class Qwen3RotaryEmbedding(nn.Module):
    """RoPE metadata placeholder; real kernel wiring belongs to a later phase."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = _head_dim(config)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_parameters = getattr(config, "rope_parameters", None)

    def forward(self, positions: object, q: object, k: object) -> tuple[object, object]:
        return q, k


class Qwen3PagedAttention(nn.Module):
    """Paged-attention 层，对齐 vLLM ``Attention`` 的所有权边界。

    分派两条路径：
    - **varlen / paged**（``attn_meta`` 非空）：``q [num_tokens,H,D]``，``k``/``v`` 为池
      L0 arena 句柄 ``[total_slots,Hkv,D]``（引擎只读，新 token 的 KV 已由池/runner
      按 ``slot_mapping`` 写入）。调 ``backend.forward_varlen``。
    - **tensors**（``attn_meta`` 为空，dev/dummy/单测）：``q``/``k``/``v`` 为
      ``[B,H,T,D]`` 原始张量。调 ``backend.forward_tensors``。

    后端与 ``block_size`` 由构造期注入（runner 经 ``build_attn_backend(name)`` 建实例
    后沿模型树下传；CPU/dev 默认 ``CpuAttentionBackend``，GPU 传 ``FlashAttn2Backend``），
    本层不 import 任何具体后端——对齐 vLLM ``Attention`` / SGLang ``RadixAttention``：
    模型层只持注入的 backend 句柄，不在 forward 内做平台判定、不 import 具体类。
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        *,
        backend: AttentionBackend | None = None,
        block_size: int = 8,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim**-0.5
        self.block_size = block_size
        self.backend = backend

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_meta: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if self.backend is None:
            raise RuntimeError(
                "Qwen3PagedAttention.backend is None — runner 未注入 attention 后端"
                "（dummy 模型 forward 不应触达本层；真实 forward 需经 ModelRunner 注入）"
            )
        if attn_meta is not None:
            return self.backend.forward_varlen(
                q,
                k,
                v,
                attn_meta,
                block_size=self.block_size,
                scale=self.scaling,
            )
        return self.backend.forward_tensors(q, k, v, is_causal=True, scale=self.scaling)


class Qwen3Attention(nn.Module):
    """Qwen3 attention shell；投影对齐 vLLM ``Qwen3Attention``。

    ``qkv_proj`` 暂用 ``ColumnParallelLinear`` 承载 packed QKV（完整
    ``QKVParallelLinear`` 含 KV head 复制逻辑，后续替换）；``o_proj`` 为
    ``RowParallelLinear``。本地 ``num_heads`` / ``num_kv_heads`` 按 TP 切分。
    """

    def __init__(self, config: Qwen3Config, layer_idx: int, *, attn_backend: AttentionBackend | None = None) -> None:
        super().__init__()
        dtype = _param_dtype(config)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = _head_dim(config)
        self.scaling = self.head_dim**-0.5

        tp_size = _tp_world_size()
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.total_num_heads} not divisible by tp_size={tp_size}"
            )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"num_key_value_heads={self.total_num_kv_heads} not divisible by tp_size={tp_size}"
                )
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            if tp_size % self.total_num_kv_heads != 0:
                raise ValueError(
                    f"tp_size={tp_size} not divisible by num_key_value_heads={self.total_num_kv_heads}"
                )
            self.num_kv_heads = 1
        # 本地 split 宽度（与 vLLM Qwen3Attention 一致）
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # packed 总输出维；ColumnParallel 按 tp 切分。KV 复制场景下与真
        # QKVParallelLinear 的分片布局可能不一致——TP>1 且 total_kv < tp 时待专类。
        qkv_out = (
            self.total_num_heads * self.head_dim
            + 2 * self.total_num_kv_heads * self.head_dim
        )
        self.qkv_proj = ColumnParallelLinearLayer(
            self.hidden_size,
            qkv_out,
            bias=bool(getattr(config, "attention_bias", False)),
            gather_output=False,
            params_dtype=dtype,
            device="meta",
        )
        self.o_proj = RowParallelLinearLayer(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            params_dtype=dtype,
            device="meta",
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.attn = Qwen3PagedAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            backend=attn_backend,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

    def forward(self, positions: object, hidden_states: object) -> object:
        return hidden_states


class Qwen3MLP(nn.Module):
    """Qwen3 MLP；对齐 vLLM ``Qwen2MLP`` 的 packed gate_up + down。

    ``gate_up_proj`` 暂用 ``ColumnParallelLinear``（``MergedColumnParallelLinear``
    后续替换）；``down_proj`` 为 ``RowParallelLinear``。
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        dtype = _param_dtype(config)
        self.gate_up_proj = ColumnParallelLinearLayer(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            device="meta",
        )
        self.down_proj = RowParallelLinearLayer(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            params_dtype=dtype,
            device="meta",
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: object) -> object:
        return hidden_states


class Qwen3Model(nn.Module):
    """Qwen3 decoder backbone skeleton.

    vLLM 的 `Qwen3Model` 继承 `Qwen2Model` 并替换 decoder layer 类型；
    lake 先钉住 module 边界和 config，真 attention/MLP 后续接 Torch/Triton。
    """

    def __init__(self, config: Qwen3Config, *, attn_backend: AttentionBackend | None = None) -> None:
        super().__init__()
        dtype = _param_dtype(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            device="meta",
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, layer_idx=i, attn_backend=attn_backend)
            for i in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.has_sliding_layers = "sliding_attention" in (getattr(config, "layer_types", None) or [])

    def forward(
        self,
        input_ids: object | None = None,
        positions: object | None = None,
        intermediate_tensors: object | None = None,
        inputs_embeds: object | None = None,
    ) -> object:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = input_ids
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        return self.norm(hidden_states)


class Qwen3DecoderLayer(nn.Module):
    """Dense Qwen3 decoder layer placeholder with stable module identity."""

    def __init__(self, config: Qwen3Config, layer_idx: int, *, attn_backend: AttentionBackend | None = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        dtype = _param_dtype(config)
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, layer_idx, attn_backend=attn_backend)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )

    def forward(self, hidden_states: object, positions: object | None = None) -> object:
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 causal LM skeleton.

    对齐 vLLM `Qwen3ForCausalLM`:顶层模型继承 `nn.Module`，持有
    `self.model = Qwen3Model(...)`，并暴露 `forward` / `compute_logits` /
    `load_weights(weights)`；dummy 由 loader 层处理。
    """

    def __init__(self, config: Qwen3Config, *, attn_backend: AttentionBackend | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, attn_backend=attn_backend)
        # vLLM 用 ParallelLMHead；未实现前用 ColumnParallel 按 vocab 维切分占位
        self.lm_head = (
            self.model.embed_tokens
            if config.tie_word_embeddings
            else ColumnParallelLinearLayer(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                gather_output=False,
                params_dtype=_param_dtype(config),
                device="meta",
            )
        )
        self.logits_processor = nn.Identity()
        self.loaded_weights: set[str] = set()
        self.loaded_dummy_weights = False

    def forward(
        self,
        input_ids: object | None = None,
        positions: object | None = None,
        intermediate_tensors: object | None = None,
        inputs_embeds: object | None = None,
    ) -> object:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: object) -> object:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, object]]) -> set[str]:
        self.loaded_weights = {name for name, _ in weights}
        return set(self.loaded_weights)


