"""
Sequence Modelling
=======
"""
# src/pacer/imitation/sequence.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, override

import numpy as np
import torch
from torch import Tensor

from pacer.base import Actions, States
from pacer.imitation.core import Collator, Criterion, Streamer
from pacer.typings import DimAction, DimState, NumPoints, npDType, torchDType

## ── Sequence Modelling ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class SequenceBatch:
    """Structural tokens for autoregressive causal attention masking."""

    returns_to_go: Tensor  # (batch, K, 1)
    states: Tensor  # (batch, K, state_dim)
    actions: Tensor  # (batch, K, action_dim)
    attention_mask: Tensor  # (batch, K) flags identifying padded steps
    weights: Tensor  # (batch,)


@dataclass(frozen=True)
class DTWindow:
    states: Tensor
    actions: Tensor
    returns_to_go: Tensor
    pad_len: int


@dataclass
class DecisionTransformerStreamer(
    Generic[NumPoints, DimState, DimAction], Streamer[DTWindow]
):
    """Extracts trajectory windows while pre-calculating total remaining returns-to-go footprints."""

    states: States[NumPoints, DimState]
    targets: Actions[NumPoints, DimAction]
    context_len: int = 20
    gamma: float = 1.0

    @override
    def __iter__(self) -> Iterator[DTWindow]:
        for i in range(self.states.length):
            demo_states = self.states[i]
            demo_actions = self.targets[i]
            T = len(demo_states)

            rewards = np.zeros(T, dtype=npDType)
            rewards[-1] = 1.0  # Mock reward as 1.0 at completion

            rtg = np.zeros(T, dtype=npDType)
            accumulated: float = 0.0
            for t in reversed(range(T)):
                accumulated = rewards[t] + self.gamma * accumulated
                rtg[t] = accumulated

            for t in range(T):
                start_idx = max(0, t - self.context_len + 1)
                yield DTWindow(
                    states=torch.tensor(
                        demo_states[start_idx : t + 1], dtype=torchDType
                    ),
                    actions=torch.tensor(
                        demo_actions[start_idx : t + 1], dtype=torchDType
                    ),
                    returns_to_go=torch.tensor(
                        rtg[start_idx : t + 1], dtype=torchDType
                    ),
                    pad_len=self.context_len - (t + 1 - start_idx),
                )


@dataclass
class SequencePaddingCollator(Collator[DTWindow, SequenceBatch]):
    """Left-pads variant-length context constraints to guarantee static causal Transformer tensor bounds."""

    context_len: int = 20

    @override
    def __call__(self, raw: DTWindow, device: torch.device) -> SequenceBatch:
        rtg_list: list[Tensor] = []
        states_list: list[Tensor] = []
        actions_list: list[Tensor] = []
        masks_list: list[Tensor] = []

        state_dim = raw.states.shape[-1]
        action_dim = raw.actions.shape[-1]
        pad_len = raw.pad_len

        # Pad structural elements with zeros if the episode just started and history < K
        padded_rtg = np.concatenate([np.zeros((pad_len,)), raw.returns_to_go])
        padded_states = np.concatenate(
            [np.zeros((pad_len, state_dim)), raw.states], axis=0
        )
        padded_actions = np.concatenate(
            [np.zeros((pad_len, action_dim)), raw.actions], axis=0
        )

        # Attention mask: 0 -> padded positions, 1 -> real historical transitions
        mask = np.concatenate([np.zeros(pad_len), np.ones(self.context_len - pad_len)])

        rtg_list.append(torch.tensor(padded_rtg, dtype=torchDType).unsqueeze(-1))
        states_list.append(torch.tensor(padded_states, dtype=torchDType))
        actions_list.append(torch.tensor(padded_actions, dtype=torchDType))
        masks_list.append(torch.tensor(mask, dtype=torchDType))

        return SequenceBatch(
            returns_to_go=torch.stack(rtg_list).to(device),
            states=torch.stack(states_list).to(device),
            actions=torch.stack(actions_list).to(device),
            attention_mask=torch.stack(masks_list).to(device),
            weights=torch.ones(1, dtype=torchDType, device=device),
        )


class MaskedActionMSECriterion(Criterion[Tensor, SequenceBatch]):
    """Calculates MSE loss evaluating only active historical action tokens via attention mask routing."""

    @override
    def __call__(
        self,
        predictions: Tensor,  # (B, K, action_dim)
        batch: SequenceBatch,
    ) -> tuple[Tensor, Tensor]:
        squared_errors = (predictions - batch.actions) ** 2
        mean_errors = squared_errors.mean(dim=-1)  # (B, K)
        masked_errors = mean_errors * batch.attention_mask  # Zero out from padding
        return masked_errors.sum(), batch.attention_mask.sum()


## ─────────────────────────────────────────────────────────────────────────────
