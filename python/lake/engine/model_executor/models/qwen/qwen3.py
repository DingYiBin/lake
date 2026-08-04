"""Qwen3 model runtime skeleton.

类名与 vLLM `Qwen3ForCausalLM` 对齐；config 由模型加载侧从 Hugging Face
config 读取后传入。Linear 对齐 vLLM：``qkv_proj`` 暂用 ColumnParallel
（``QKVParallelLinear`` 后续替换）、``gate_up_proj`` 用 MergedColumnParallel、
``o_proj``/``down_proj`` 用 RowParallel；``lm_head`` 未 tie 时用 ColumnParallel
（占位 ParallelLMHead）。

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
    MergedColumnParallelLinearLayer,
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
    """RMSNorm parameter shell matching Qwen3/vLLM naming.

    C16a：落实真实 RMSNorm（对齐 Transformers ``Qwen3RMSNorm.forward``——fp32 方差 +
    rsqrt，再乘权重）。``residual`` 可选：先加残差再归一化（对齐 vLLM RMSNorm 的
    残差融合入口），无残差时直接归一化。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, device="meta", dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            hidden_states = hidden_states + residual
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3RotaryEmbedding(nn.Module):
    """RoPE；C16a 落实真实旋转位置编码。

    对齐 Transformers ``Qwen3RotaryEmbedding`` + ``apply_rotary_pos_emb``：现场由
    ``rope_theta`` + ``head_dim`` 算 ``inv_freq``（不依赖 ``meta`` buffer，避免
    物化时 buffer 残留 meta），对 ``positions [T]`` 产 cos/sin，``rotate_half``
    应用到 q/k。lake KV 归池、模型 API 不持 KV 生命周期，故此处只旋转 q/k 张量。
    """

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = _head_dim(config)
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_parameters = getattr(config, "rope_parameters", None)
        rope_theta = 10000.0
        if self.rope_parameters is not None:
            rope_theta = float(self.rope_parameters.get("rope_theta", 10000.0))
        self.rope_theta = rope_theta

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inv_freq [head_dim/2]
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=q.device) / self.head_dim)
        )
        # freqs [T, head_dim/2] -> emb [T, head_dim]
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        # q/k [..., head_dim]；cos/sin [T, head_dim] -> 广播到 head 维
        # 期望 q/k 形如 [T, H, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed


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

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """C16a/C16b：真实 attention forward。

        paged 路径（forward context 携 ``attn_meta`` + per-layer KV arena）：``qkv_proj``
        → 拆 q/k/v → ``q_norm``/``k_norm`` → RoPE → 写新 k/v 到 arena（按 ``slot_mapping``）
        → ``forward_varlen`` 读 paged KV → ``o_proj``。非分页 dummy 路径（无 context）：
        RoPE → ``forward_tensors``（B=1、全序列 causal）→ ``o_proj``。
        """
        T = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)  # [T, q_size + 2*kv_size]（TP=1 全宽）
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(T, self.num_heads, self.head_dim)
        k = k.view(T, self.num_kv_heads, self.head_dim)
        v = v.view(T, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        from lake.engine.model_executor.layers.attentions.context import get_forward_context
        ctx = get_forward_context()
        if (
            ctx is not None
            and ctx.attn_meta is not None
            and ctx.kv_caches is not None
        ):
            meta = ctx.attn_meta
            k_cache, v_cache = ctx.kv_caches[self.layer_idx]
            slot_mapping = meta.slot_mapping  # [T]（effective，pad 槽为 -1）
            valid = slot_mapping >= 0
            if valid.any():
                k_cache.index_copy_(0, slot_mapping[valid].long(), k[valid])
                v_cache.index_copy_(0, slot_mapping[valid].long(), v[valid])
            # forward_varlen：q [num_tokens, H, D]，k/v 为 arena 句柄
            out = self.attn(q, k_cache, v_cache, attn_meta=meta)  # [num_tokens, H, D]
            attn_out = out.reshape(T, self.num_heads * self.head_dim)
            return self.o_proj(attn_out)

        # 非分页 dummy 路径（C16a）：forward_tensors 要 [B, H, T, D]：B=1
        q_bt = q.unsqueeze(0).permute(0, 2, 1, 3)  # [1, H, T, D]
        k_bt = k.unsqueeze(0).permute(0, 2, 1, 3)
        v_bt = v.unsqueeze(0).permute(0, 2, 1, 3)
        attn_out = self.attn(q_bt, k_bt, v_bt)  # [1, H, T, D]
        # [1, H, T, D] -> [T, H*D]
        attn_out = attn_out.squeeze(0).movedim(0, 1).reshape(T, self.num_heads * self.head_dim)
        return self.o_proj(attn_out)


class Qwen3MLP(nn.Module):
    """Qwen3 MLP；对齐 vLLM ``Qwen2MLP`` 的 packed gate_up + down。"""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        dtype = _param_dtype(config)
        inter = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinearLayer(
            config.hidden_size,
            [inter, inter],
            bias=False,
            gather_output=False,
            params_dtype=dtype,
            device="meta",
        )
        self.down_proj = RowParallelLinearLayer(
            inter,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            params_dtype=dtype,
            device="meta",
        )
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """C16a：真实 MLP（对齐 vLLM ``Qwen2MLP``：``down(act(gate)*up)``）。"""
        gate_up = self.gate_up_proj(hidden_states)  # [T, 2*intermediate]
        inter = gate_up.shape[-1] // 2
        gate, up = gate_up.split([inter, inter], dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


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
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """C16a：真实 backbone forward（embed → layers → norm）。返回 hidden [T, hidden]。"""
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
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

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """C16a：真实 decoder layer（双残差 + 双 RMSNorm + attn + mlp）。

        对齐 Transformers ``Qwen3DecoderLayer``：``h = h + attn(norm1(h))``；
        ``h = h + mlp(norm2(h))``。RMSNorm 不融合残差（残差在层内显式加），与
        vLLM RMSNorm 的残差融合入口区别——lake CPU 路径先求简。
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


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
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """C16a：返回 backbone hidden [T, hidden]（不在此处算 logits，由
        ``compute_logits`` 单独取，便于 runner 只算末位 token）。"""
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """C16a：``hidden [T, hidden]`` → ``lm_head`` → ``[T, vocab]``。

        tie 时 ``lm_head is embed_tokens``：用 ``F.linear(hidden, embed.weight)``；
        否则 ``lm_head`` 为 ``ColumnParallelLinear``（``gather_output=False``，TP=1
        即全宽 vocab）。
        """
        if self.lm_head is self.model.embed_tokens:
            return torch.nn.functional.linear(hidden_states, self.model.embed_tokens.weight)
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, object]]) -> set[str]:
        self.loaded_weights = {name for name, _ in weights}
        return set(self.loaded_weights)


