"""Tiny offline model config for unit tests.

对齐 vLLM ``dummy_hf_overrides`` / SGLang ``--json-model-override-args``：
缩 ``num_hidden_layers`` 等，配合 ``load_format=dummy`` 避免下权重、避免大模型树。
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import Qwen3Config

from lake.engine.model_runner import ModelRunner


def tiny_qwen3_config(**overrides: Any) -> Qwen3Config:
    """1-layer Qwen3 config；不触网，供 registry + DummyModelLoader 路径。"""
    kwargs: dict[str, Any] = {
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 256,
        "max_position_embeddings": 512,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": True,
    }
    kwargs.update(overrides)
    return Qwen3Config(**kwargs)


def make_runner(
    pool: Any,
    *,
    load: bool = True,
    model_config: Optional[Any] = None,
    **runner_kwargs: Any,
) -> ModelRunner:
    """Construct a ModelRunner with tiny config; optionally ``load_format=dummy``."""
    runner = ModelRunner(
        pool,
        model_config=model_config or tiny_qwen3_config(),
        **runner_kwargs,
    )
    if load:
        runner.load_model(model_path="tiny-qwen3", load_format="dummy")
    return runner
