"""Sampler reference path.

参考 vLLM/SGLang sampler 的边界：sampler 是 ``nn.Module``，先按 sampling
params 变换 logits，再从最终分布采样。当前只覆盖 temperature / top_k / top_p，
后续可在 ``forward`` 内按 device/backend 分支到 Triton/CUDA/NPU kernel，或叠加
penalty / min_p / logits processor。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

_GREEDY_EPS = 1e-6


@dataclass(frozen=True)
class SamplingMetadata:
    """Batched sampling parameters.

    ``cum_num_sampling_tokens`` has length ``num_reqs + 1``. The range
    ``[cum[i], cum[i + 1])`` maps logits rows to request ``i`` and reuses that
    request's temperature / top-k / top-p parameters.
    """

    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    cum_num_sampling_tokens: torch.Tensor
    sampling_seeds: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    all_greedy: bool = False
    all_random: bool = False

    @classmethod
    def from_lists(
        cls,
        *,
        temperatures: Sequence[float],
        top_ks: Sequence[int],
        top_ps: Sequence[float],
        cum_num_sampling_tokens: Sequence[int],
        sampling_seeds: Sequence[int | None] | None = None,
        positions: Sequence[int] | None = None,
        device: torch.device | str | None = None,
        all_greedy: bool | None = None,
        all_random: bool | None = None,
    ) -> "SamplingMetadata":
        temps = list(temperatures)
        cum = list(cum_num_sampling_tokens)
        seed_tensor = None
        if sampling_seeds is not None:
            seed_tensor = torch.tensor(
                [42 if s is None else int(s) for s in sampling_seeds],
                dtype=torch.int64,
                device=device,
            )
        pos_tensor = None
        if positions is not None:
            pos_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
        elif seed_tensor is not None:
            pos_tensor = torch.arange(cum[-1], dtype=torch.int64, device=device)
        if all_greedy is None:
            all_greedy = all(t <= _GREEDY_EPS for t in temps)
        if all_random is None:
            all_random = all(t > _GREEDY_EPS for t in temps)
        return cls(
            temperatures=torch.tensor(temps, dtype=torch.float32, device=device),
            top_ks=torch.tensor(top_ks, dtype=torch.int32, device=device),
            top_ps=torch.tensor(top_ps, dtype=torch.float32, device=device),
            cum_num_sampling_tokens=torch.tensor(
                cum, dtype=torch.int32, device=device
            ),
            sampling_seeds=seed_tensor,
            positions=pos_tensor,
            all_greedy=all_greedy,
            all_random=all_random,
        )

    @property
    def num_reqs(self) -> int:
        return int(self.cum_num_sampling_tokens.numel() - 1)


class Sampler(nn.Module):
    """Reference sampler implemented with torch ops."""

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample token ids from ``[num_sampling_tokens, vocab]`` logits.

        Per-request parameters are broadcast over row ranges described by
        ``sampling_metadata.cum_num_sampling_tokens``.
        """

        scores = logits.to(dtype=torch.float32)

        temperatures, top_ks, top_ps = _expand_params(scores, sampling_metadata)

        if sampling_metadata.all_greedy:
            return torch.argmax(scores, dim=-1)

        safe_temperatures = torch.where(
            temperatures <= _GREEDY_EPS,
            torch.ones_like(temperatures),
            temperatures,
        )
        random_scores = scores / safe_temperatures.unsqueeze(-1)
        random_scores = _apply_top_k(random_scores, top_ks)
        random_scores = _apply_top_p(random_scores, top_ps)
        probs = torch.softmax(random_scores, dim=-1)
        if sampling_metadata.sampling_seeds is None:
            random_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            seeds = _expand_sampling_seeds(scores, sampling_metadata)
            positions = sampling_metadata.positions.to(
                device=scores.device,
                dtype=torch.long,
            )
            random_sampled = _deterministic_sample(probs, seeds, positions)

        if sampling_metadata.all_random:
            return random_sampled

        greedy_sampled = torch.argmax(scores, dim=-1)
        return torch.where(temperatures <= _GREEDY_EPS, greedy_sampled, random_sampled)


def _expand_params(
    scores: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cum = sampling_metadata.cum_num_sampling_tokens.to(
        device=scores.device,
        dtype=torch.long,
    )
    counts = cum[1:] - cum[:-1]
    output_size = scores.shape[0]
    temperatures = sampling_metadata.temperatures.to(
        device=scores.device, dtype=torch.float32
    ).repeat_interleave(counts, output_size=output_size)
    top_ks = sampling_metadata.top_ks.to(
        device=scores.device, dtype=torch.long
    ).repeat_interleave(counts, output_size=output_size)
    top_ps = sampling_metadata.top_ps.to(
        device=scores.device, dtype=torch.float32
    ).repeat_interleave(counts, output_size=output_size)
    return temperatures, top_ks, top_ps


def _expand_sampling_seeds(
    scores: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    cum = sampling_metadata.cum_num_sampling_tokens.to(
        device=scores.device,
        dtype=torch.long,
    )
    counts = cum[1:] - cum[:-1]
    return sampling_metadata.sampling_seeds.to(
        device=scores.device, dtype=torch.long
    ).repeat_interleave(counts, output_size=scores.shape[0])


def _apply_top_k(scores: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    vocab_size = scores.shape[-1]
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    ranks = torch.arange(vocab_size, device=scores.device).unsqueeze(0)
    keep = ranks < top_ks.unsqueeze(-1)
    sorted_scores = sorted_scores.masked_fill(~keep, float("-inf"))
    filtered = torch.full_like(scores, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_scores)
    return filtered


def _deterministic_sample(
    probs: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    logprobs = torch.log(probs.to(torch.float64))
    col_indices = torch.arange(probs.shape[-1], device=probs.device, dtype=torch.long)
    uniform = _uniform_from_seed(seeds, positions, col_indices)
    tiny = torch.finfo(torch.float64).tiny
    uniform = torch.clamp(
        uniform,
        min=tiny,
        max=1.0 - torch.finfo(torch.float64).eps,
    )
    gumbel = -torch.log(-torch.log(uniform))
    return torch.argmax(logprobs + gumbel, dim=-1)


def _uniform_from_seed(
    seeds: torch.Tensor,
    positions: torch.Tensor,
    col_indices: torch.Tensor,
) -> torch.Tensor:
    x = seeds.unsqueeze(-1).to(torch.long)
    x = x ^ (positions.unsqueeze(-1).to(torch.long) * 0x9E3779B1)
    x = x ^ (col_indices.unsqueeze(0).to(torch.long) * 0x85EBCA77)
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    x = x ^ (x >> 16)
    x = torch.bitwise_and(x, 0xFFFFFFFF)
    return (x.to(torch.float64) + 1.0) / (float(2**32) + 1.0)


def _apply_top_p(scores: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_scores, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_ps.unsqueeze(-1)
    shifted = remove.clone()
    shifted[:, 1:] = remove[:, :-1]
    shifted[:, 0] = False
    remove = shifted
    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(scores, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_scores)
    return filtered
