"""Sampling hooks independent of model backend."""

from __future__ import annotations

import torch

from lake.engine.agents.memory import InMemoryAgent
from lake.engine.pool_iface import PoolIface
from lake.engine.sample.grammar import apply_token_bitmask
from lake.engine.sample.greedy import greedy_sample
from lake.engine.sample.reject import RejectionSampler, SpeculativeMetadata
from lake.engine.sample.sampler import Sampler, SamplingMetadata
from lake.runtime.req import Req
from lake.runtime.scheduler_output import (
    ForwardMode,
    GrammarOutput,
    SamplingParams,
    SpeculativeParameters,
    SchedulerOutput,
    TOP_K_ALL,
)
from lake.testing import make_runner


def test_greedy_sample() -> None:
    assert greedy_sample([0.1, 0.9, 0.2]) == 1


def test_apply_token_bitmask() -> None:
    masked = apply_token_bitmask([0.1, 0.9, 0.2], [True, False, True])
    assert greedy_sample(masked) == 2


def test_sampler_greedy_temperature_zero() -> None:
    metadata = SamplingMetadata.from_lists(
        temperatures=[0.0],
        top_ks=[TOP_K_ALL],
        top_ps=[1.0],
    )
    logits = torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float32)
    assert Sampler()(logits, metadata).tolist() == [1]


def test_sampler_top_k_one() -> None:
    metadata = SamplingMetadata.from_lists(
        temperatures=[1.0],
        top_ks=[1],
        top_ps=[1.0],
    )
    logits = torch.tensor([[0.1, 0.9, 0.2]], dtype=torch.float32)
    assert Sampler()(logits, metadata).tolist() == [1]


def test_sampler_top_p_keeps_min_prefix() -> None:
    metadata = SamplingMetadata.from_lists(
        temperatures=[1.0],
        top_ks=[TOP_K_ALL],
        top_ps=[0.5],
    )
    logits = torch.tensor([[10.0, 9.0, 0.0]], dtype=torch.float32)
    assert Sampler()(logits, metadata).tolist() == [0]


def test_sampler_uses_params_by_row() -> None:
    metadata = SamplingMetadata.from_lists(
        temperatures=[0.0, 1.0],
        top_ks=[TOP_K_ALL, 1],
        top_ps=[1.0, 1.0],
    )
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.3, 0.2, 0.8],
        ],
        dtype=torch.float32,
    )
    assert Sampler()(logits, metadata).tolist() == [1, 2]


def test_sampler_seeded_sampling_is_deterministic() -> None:
    metadata = SamplingMetadata.from_lists(
        temperatures=[1.0, 1.0],
        top_ks=[TOP_K_ALL, TOP_K_ALL],
        top_ps=[1.0, 1.0],
        sampling_seeds=[1234, 1234],
        positions=[7, 8],
    )
    logits = torch.zeros((2, 8), dtype=torch.float32)

    first = Sampler()(logits, metadata)
    second = Sampler()(logits, metadata)

    assert first.tolist() == second.tolist()


def test_rejection_sampler_accepts_all_drafts_and_bonus() -> None:
    sampler = RejectionSampler(SpeculativeParameters(num_speculative_tokens=2))
    target_probs = torch.tensor(
        [
            [
                [0.1, 0.8, 0.1, 0.0],
                [0.1, 0.1, 0.8, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    draft_probs = torch.tensor(
        [
            [
                [0.1, 0.8, 0.1, 0.0],
                [0.1, 0.1, 0.8, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    out = sampler(
        _probs_to_logits(target_probs),
        SpeculativeMetadata(
            draft_token_ids=torch.tensor([[1, 2]], dtype=torch.long),
            draft_probs=draft_probs,
            uniform_samples=torch.zeros((1, 2), dtype=torch.float32),
            uniform_samples_for_final=torch.tensor([0.5], dtype=torch.float32),
        ),
        SamplingMetadata.from_lists(
            temperatures=[1.0],
            top_ks=[TOP_K_ALL],
            top_ps=[1.0],
        ),
    )

    assert out.token_ids.tolist() == [[1, 2, 3]]
    assert out.num_accepted.tolist() == [2]
    assert out.num_sampled.tolist() == [3]


def test_rejection_sampler_samples_residual_on_reject() -> None:
    sampler = RejectionSampler(SpeculativeParameters(num_speculative_tokens=2))
    target_probs = torch.tensor(
        [
            [
                [0.2, 0.7, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ]
        ],
        dtype=torch.float32,
    )
    draft_probs = torch.tensor(
        [
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ]
        ],
        dtype=torch.float32,
    )

    out = sampler(
        _probs_to_logits(target_probs),
        SpeculativeMetadata(
            draft_token_ids=torch.tensor([[0, 1]], dtype=torch.long),
            draft_probs=draft_probs,
            uniform_samples=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
            uniform_samples_for_final=torch.tensor([0.5], dtype=torch.float32),
        ),
        SamplingMetadata.from_lists(
            temperatures=[1.0],
            top_ks=[TOP_K_ALL],
            top_ps=[1.0],
        ),
    )

    assert out.token_ids.tolist() == [[1, -1, -1]]
    assert out.num_accepted.tolist() == [0]
    assert out.num_sampled.tolist() == [1]


def test_sample_tokens_uses_grammar_bitmask() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    runner = make_runner(pool, load=False)
    req = Req(
        req_id="g1",
        served_model_name="model",
        prompt_token_ids=[0],
        sampling_params=SamplingParams(
            max_new_tokens=1,
            temperature=0.0,
            structured_output="json",
        ),
    )
    output = SchedulerOutput(
        step_id=1,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={"g1": 1},
        total_num_scheduled_tokens=1,
        grammar_output=GrammarOutput(
            req_ids=["g1"],
            token_bitmask_by_req={"g1": [True, False, True]},
        ),
        has_structured_output=True,
    )
    sampled, _ = runner.sample_tokens(output, {"g1": req}, {"g1": [0.1, 0.9, 0.2]})
    assert sampled == {"g1": [2]}


def test_sample_tokens_can_defer_structured_output() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    runner = make_runner(pool, load=False)
    req = Req(
        req_id="g2",
        served_model_name="model",
        prompt_token_ids=[0],
        sampling_params=SamplingParams(
            max_new_tokens=1,
            temperature=0.0,
            structured_output="json",
        ),
    )
    output = SchedulerOutput(
        step_id=2,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={"g2": 1},
        total_num_scheduled_tokens=1,
        grammar_output=GrammarOutput(
            req_ids=["g2"],
            deferred_req_ids=["g2"],
            reason="waiting for prior token",
        ),
        has_structured_output=True,
    )
    sampled, _ = runner.sample_tokens(output, {"g2": req}, {"g2": [0.1, 0.9, 0.2]})
    assert sampled == {}


def _probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(probs, min=torch.finfo(probs.dtype).tiny))
