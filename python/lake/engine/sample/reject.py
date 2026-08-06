"""链式拒绝采样（Leviathan / SGLang chain_speculative_sampling 简化版）。

参考:SGLang `speculative/reject_sampling.py::chain_speculative_sampling_triton`；
vLLM RejectionSampler。C4：贪心 target 与 draft 逐位比对，遇分歧停并补 bonus。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
import torch.nn as nn

from lake.engine.sample.sampler import (
    _GREEDY_EPS,
    _apply_top_k,
    _apply_top_p,
    SamplingMetadata,
)
from lake.runtime.scheduler_output import SpeculativeParameters


_REJECTED_TOKEN_ID = -1


@dataclass(frozen=True)
class RejectionSamplerOutput:
    token_ids: torch.Tensor
    num_accepted: torch.Tensor
    num_sampled: torch.Tensor
    accepted_mask: torch.Tensor


@dataclass(frozen=True)
class SpeculativeMetadata:
    draft_token_ids: torch.Tensor
    draft_probs: torch.Tensor
    uniform_samples: torch.Tensor | None = None
    uniform_samples_for_final: torch.Tensor | None = None


class RejectionSampler(nn.Module):
    """Pure torch speculative rejection sampler.

    Inputs are row-aligned by request. ``logits`` are target verify logits and
    have one extra row per request for the final bonus/residual sample.
    """

    def __init__(self, speculative_parameters: SpeculativeParameters) -> None:
        super().__init__()
        self.speculative_parameters = speculative_parameters

    def forward(
        self,
        logits: torch.Tensor,
        speculative_metadata: SpeculativeMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> RejectionSamplerOutput:
        num_steps = self.speculative_parameters.num_speculative_tokens
        target_probs = _target_probs_from_logits(
            logits,
            sampling_metadata,
            num_steps,
        )
        draft_probs = speculative_metadata.draft_probs.to(
            device=logits.device,
            dtype=torch.float32,
        )

        batch_size = speculative_metadata.draft_token_ids.shape[0]
        device = logits.device
        uniform_samples = speculative_metadata.uniform_samples
        if uniform_samples is None:
            uniform_samples = torch.rand(
                batch_size,
                num_steps,
                device=device,
                dtype=torch.float32,
            )
        else:
            uniform_samples = uniform_samples.to(device=device, dtype=torch.float32)
        uniform_samples_for_final = speculative_metadata.uniform_samples_for_final
        if uniform_samples_for_final is None:
            uniform_samples_for_final = torch.rand(
                batch_size,
                device=device,
                dtype=torch.float32,
            )
        else:
            uniform_samples_for_final = uniform_samples_for_final.to(
                device=device,
                dtype=torch.float32,
            )

        draft_tokens = speculative_metadata.draft_token_ids.to(
            device=device,
            dtype=torch.long,
        )
        sampled_probs = _gather_draft_token_probs(
            target_probs[:, :num_steps, :],
            draft_probs,
            draft_tokens,
        )
        target_on_draft, draft_on_draft = sampled_probs
        accepted = uniform_samples * draft_on_draft <= target_on_draft
        accepted_mask = torch.cumprod(accepted.to(torch.int32), dim=1).to(torch.bool)
        num_accepted = torch.sum(accepted_mask.to(torch.long), dim=1)

        final_probs = _final_distribution(target_probs, draft_probs, num_accepted)
        final_token_ids = _sample_from_probs(final_probs, uniform_samples_for_final)

        token_ids = torch.full(
            (batch_size, num_steps + 1),
            _REJECTED_TOKEN_ID,
            dtype=torch.long,
            device=device,
        )
        token_ids[:, :num_steps] = torch.where(
            accepted_mask,
            draft_tokens,
            torch.full_like(draft_tokens, _REJECTED_TOKEN_ID),
        )
        token_ids.scatter_(
            1,
            num_accepted.unsqueeze(-1),
            final_token_ids.unsqueeze(-1),
        )

        return RejectionSamplerOutput(
            token_ids=token_ids,
            num_accepted=num_accepted,
            num_sampled=num_accepted + 1,
            accepted_mask=accepted_mask,
        )


def chain_reject_sample(
    context: Sequence[int],
    draft_tokens: Sequence[int],
    target_greedy: Callable[[Sequence[int]], int],
) -> List[int]:
    """返回本步接受的 token 列表（含分歧处的 target bonus，长度 ∈ [1, len(draft)+1]）。

    约定：对每个 draft[i]，用 target 在 context+accepted 上的 greedy 与之比较；
    全中则再追加 1 个 bonus greedy token。
    """
    accepted: List[int] = []
    ctx = list(context)
    for d in draft_tokens:
        t = target_greedy(ctx)
        if t != int(d):
            accepted.append(t)
            return accepted
        accepted.append(int(d))
        ctx.append(int(d))
    # 全部命中 → bonus
    accepted.append(target_greedy(ctx))
    return accepted


def _target_probs_from_logits(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    num_steps: int,
) -> torch.Tensor:
    scores = logits.to(dtype=torch.float32)
    batch_size, rows_per_req, vocab_size = scores.shape
    flat_scores = scores.reshape(batch_size * rows_per_req, vocab_size)
    output_size = batch_size * (num_steps + 1)
    temperatures = sampling_metadata.temperatures.to(
        device=scores.device,
        dtype=torch.float32,
    ).repeat_interleave(num_steps + 1, output_size=output_size)
    top_ks = sampling_metadata.top_ks.to(
        device=scores.device,
        dtype=torch.long,
    ).repeat_interleave(num_steps + 1, output_size=output_size)
    top_ps = sampling_metadata.top_ps.to(
        device=scores.device,
        dtype=torch.float32,
    ).repeat_interleave(num_steps + 1, output_size=output_size)

    safe_temperatures = torch.where(
        temperatures <= _GREEDY_EPS,
        torch.ones_like(temperatures),
        temperatures,
    )
    random_scores = flat_scores / safe_temperatures.unsqueeze(-1)
    random_scores = _apply_top_k(random_scores, top_ks)
    random_scores = _apply_top_p(random_scores, top_ps)
    probs = torch.softmax(random_scores, dim=-1)

    greedy_token_ids = torch.argmax(flat_scores, dim=-1)
    greedy_probs = torch.zeros_like(probs)
    greedy_probs.scatter_(1, greedy_token_ids.unsqueeze(-1), 1.0)
    probs = torch.where(
        (temperatures <= _GREEDY_EPS).unsqueeze(-1),
        greedy_probs,
        probs,
    )
    return probs.reshape(batch_size, rows_per_req, vocab_size)


def _gather_draft_token_probs(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gather_index = draft_token_ids.unsqueeze(-1)
    target_on_draft = torch.gather(
        target_probs,
        dim=-1,
        index=gather_index,
    ).squeeze(-1)
    draft_on_draft = torch.gather(
        draft_probs,
        dim=-1,
        index=gather_index,
    ).squeeze(-1)
    return target_on_draft, draft_on_draft


def _final_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    num_accepted: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, vocab_size = target_probs.shape
    padded_draft = torch.cat(
        [
            draft_probs,
            torch.zeros(
                batch_size,
                1,
                vocab_size,
                device=draft_probs.device,
                dtype=draft_probs.dtype,
            ),
        ],
        dim=1,
    )
    gather_index = num_accepted.view(-1, 1, 1).expand(-1, 1, vocab_size)
    target = torch.gather(target_probs, dim=1, index=gather_index).squeeze(1)
    draft = torch.gather(padded_draft, dim=1, index=gather_index).squeeze(1)
    residual = torch.clamp(target - draft, min=0.0)
    residual_mass = torch.sum(residual, dim=-1, keepdim=True)
    normalized_residual = residual / torch.clamp(
        residual_mass,
        min=torch.finfo(residual.dtype).tiny,
    )
    return torch.where(residual_mass > 0.0, normalized_residual, target)


def _sample_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    token_ids = torch.sum(
        cdf < uniform_samples.unsqueeze(-1),
        dim=-1,
        dtype=torch.long,
    )
    return torch.clamp(token_ids, max=probs.shape[-1] - 1)
