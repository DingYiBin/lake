"""Model loader skeletons.

对齐 vLLM: loader 统一创建模型，具体 load format 只实现权重加载差异。
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Generic, Literal, TypeVar

import torch
import torch.nn as nn


TModel = TypeVar("TModel")
TConfig = TypeVar("TConfig")
LoadFormat = Literal["dummy", "hf"]


class BaseModelLoader(Generic[TModel, TConfig]):
    """Base loader: create model first, then delegate weight loading."""

    def load_model(
        self,
        model_cls: type[TModel],
        config: TConfig,
        *,
        attn_backend: object | None = None,
    ) -> TModel:
        # 仅当模型类接受 ``attn_backend`` 时注入（Qwen3ForCausalLM 接受；测试/自定义
        # 模型可能不接受，回退到 ``model_cls(config)``）——对齐 vLLM 对可选构造 kwarg
        # 的兼容处理，避免强制所有模型类改签名。
        if "attn_backend" in inspect.signature(model_cls).parameters:
            model = model_cls(config, attn_backend=attn_backend)
        else:
            model = model_cls(config)
        loaded = self.load_weights(model)
        if loaded is not None:
            setattr(model, "loaded_weights", loaded)
        return model

    def load_weights(self, model: TModel) -> set[str] | None:
        raise NotImplementedError


class DummyModelLoader(BaseModelLoader[TModel, TConfig]):
    """Loader that initializes fake weights through the model's load_weights API."""

    def __init__(self, weight_names: Iterable[str] | None = None) -> None:
        self._weight_names = tuple(weight_names) if weight_names is not None else None

    def load_weights(self, model: TModel) -> set[str]:
        load_weights = getattr(model, "load_weights")
        loaded = load_weights(self.iter_dummy_weights(model))
        setattr(model, "loaded_dummy_weights", True)
        return loaded

    def iter_dummy_weights(self, model: TModel) -> Iterable[tuple[str, object]]:
        names = self._weight_names
        if names is None:
            state_dict = getattr(model, "state_dict")
            names = tuple(state_dict().keys())
        for name in names:
            yield name, object()


class DefaultModelLoader(BaseModelLoader[TModel, TConfig]):
    """Loader for real weight files.

    The file iterator is intentionally not implemented yet; this class fixes the
    boundary that future safetensors/bin loading will fill in.
    """

    def __init__(self, model_path: str, revision: str = "") -> None:
        self.model_path = model_path
        self.revision = revision

    def load_weights(self, model: TModel) -> set[str] | None:
        raise NotImplementedError("real weight loading is not implemented yet")


def get_model_loader(
    load_format: LoadFormat,
    *,
    model_path: str = "",
    revision: str = "",
) -> BaseModelLoader[TModel, TConfig]:
    if load_format == "dummy":
        return DummyModelLoader()
    if load_format == "hf":
        return DefaultModelLoader(model_path=model_path, revision=revision)
    raise ValueError(f"unsupported load_format={load_format!r}")


def materialize_model(
    model: TModel,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    seed: int = 0,
) -> TModel:
    """把 ``meta`` device 上的模型物化到真实 device 并做小随机 init。

    C16a：dummy 路径需要真实权重才能跑 forward。模型构造在 ``device="meta"``
    （避免骨架阶段分配 0.6B 权重）；本函数做 ``to_empty`` 后按角色 init：
    RMSNorm 权重置 1（保 identity-ish）、embedding/linear 小 normal、bias 置 0，
    使 forward 产有限非退化 logits。仅对 ``nn.Module`` 生效；非 Module（自定义/
    测试桩）原样返回。对齐 vLLM ``to_empty`` + ``materialize`` 的 dummy 物化语义。
    """
    if not isinstance(model, nn.Module):
        return model
    has_meta = any(p.device.type == "meta" for p in model.parameters())
    has_meta = has_meta or any(
        getattr(b, "device", torch.device("cpu")).type == "meta"
        for b in model.buffers()
    )
    if has_meta:
        model.to_empty(device=device)
    if dtype is not None:
        model.to(dtype)
    gen = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02, generator=gen)
        elif hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
            # RMSNorm-like：权重置 1，forward 近似 identity
            nn.init.ones_(module.weight)
        elif hasattr(module, "weight") and isinstance(getattr(module, "weight", None), nn.Parameter):
            nn.init.normal_(module.weight, std=0.02, generator=gen)
            bias = getattr(module, "bias", None)
            if bias is not None:
                nn.init.zeros_(bias)
    return model
