"""
Diffusion Models
=======
"""
# src/pacer/imitation/diffusion.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, override

import torch
from torch import Tensor

from pacer.base import Actions, States
from pacer.imitation.core import Collator, Criterion, Streamer
from pacer.typings import DimAction, DimState, FloatLike, NumPoints, Vector, torchDType

## ── Diffusion ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiffusionBatch:
    """Noise-perturbed trajectory contexts."""

    state_context: Tensor  # (batch, context_len, state_dim)
    noisy_action_horizon: Tensor  # (batch, horizon_len, action_dim)
    noise_steps: Tensor  # (batch,) discrete steps [0, N-1]
    target_noise: Tensor  # (batch, horizon_len, action_dim)
    weights: Tensor  # (batch,)


@dataclass(frozen=True)
class DiffusionWindow:
    state_context: Tensor
    action_horizon: Tensor
    weight: FloatLike = 1.0


@dataclass
class DiffusionTrajectoryStreamer(
    Generic[NumPoints, DimState, DimAction], Streamer[DiffusionWindow]
):
    """Slices demonstrations into overlapping sequence sub-windows for context and future horizons."""

    states: States[NumPoints, DimState]
    targets: Actions[NumPoints, DimAction]
    context_len: int = 2
    horizon_len: int = 16
    weights: Vector[NumPoints] | None = None

    @override
    def __iter__(self) -> Iterator[DiffusionWindow]:
        for i in range(self.states.length):
            demo_states = self.states[i]  # (T, state_dim)
            demo_actions = self.targets[i]  # (T, action_dim)
            demo_weight = self.weights[i] if self.weights is not None else None
            T = len(demo_states)

            for t in range(self.context_len - 1, T - self.horizon_len):
                yield DiffusionWindow(
                    state_context=torch.tensor(
                        demo_states[t - self.context_len + 1 : t + 1], dtype=torchDType
                    ),
                    action_horizon=torch.tensor(
                        demo_actions[t : t + self.horizon_len], dtype=torchDType
                    ),
                    weight=demo_weight if demo_weight is not None else 1.0,
                )


@dataclass
class DiffusionBatchCollator(Collator[DiffusionWindow, DiffusionBatch]):
    """Applies forward diffusion variance scheduling to corrupt actions with Gaussian noise."""

    num_diffusion_steps: int = 100

    def __post_init__(self) -> None:
        # Fixed linear variance schedule
        self.betas: Tensor = torch.linspace(1e-4, 0.02, self.num_diffusion_steps)
        self.alphas: Tensor = 1.0 - self.betas
        self.alphas_cumprod: Tensor = torch.cumprod(self.alphas, dim=0)

    @override
    def __call__(self, raw: DiffusionWindow, device: torch.device) -> DiffusionBatch:
        states_list: list[Tensor] = [raw.state_context]
        actions_list: list[Tensor] = [raw.action_horizon]
        weights_list: list[Tensor] = [torch.tensor(raw.weight)]

        state_context = torch.stack(states_list).to(
            device
        )  # (B, context_len, state_dim)
        clean_actions = torch.stack(actions_list).to(
            device
        )  # (B, horizon_len, action_dim)
        weights = torch.tensor(weights_list, dtype=torchDType, device=device)

        B = state_context.size(0)
        t_steps = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
        noise = torch.randn_like(clean_actions)  # (B, horizon_len, action_dim)

        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t_steps]).view(B, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod[t_steps]
        ).view(B, 1, 1)

        noisy_actions = (
            sqrt_alpha_cumprod * clean_actions + sqrt_one_minus_alpha_cumprod * noise
        )

        return DiffusionBatch(
            state_context=state_context,
            noisy_action_horizon=noisy_actions,
            noise_steps=t_steps,
            target_noise=noise,
            weights=weights,
        )


class DiffusionMSECriterion(Criterion[Tensor, DiffusionBatch]):
    """Evaluates how perfectly the network can isolate and extract injected noise vectors."""

    @override
    def __call__(
        self,
        predictions: Tensor,  # Noise estimates; (B, horizon_len, action_dim)
        batch: DiffusionBatch,
    ) -> tuple[Tensor, Tensor]:
        square_error = (predictions - batch.target_noise) ** 2
        per_sample_loss = square_error.mean(dim=[-1, -2]) * batch.weights
        return per_sample_loss.sum(), batch.weights.sum()


## ─────────────────────────────────────────────────────────────────────────────
