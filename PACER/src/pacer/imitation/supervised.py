"""
Supervised Learning
=======
"""
# src/pacer/imitation/supervised.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from typingkit.core import RuntimeGeneric

from pacer.base import Actions, States
from pacer.imitation.core import (
    BatchT,
    Collator,
    Criterion,
    Evaluator,
    Hook,
    PolicyT,
    PredictionT,
    RawDataT,
    StepExecutor,
    Streamer,
    Workflow,
)
from pacer.typings import DimAction, DimState, NumPoints, Vector, torchDType
from pacer.utils import EPS, SEED, set_seed

## ── Supervised Learning ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class GMMOutputs:
    pi: Tensor  # (batch, n_components)
    mu: Tensor  # (batch, n_components, action_dim)
    sigma: Tensor  # (batch, n_components, action_dim)


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SupervisedWorkflow(Workflow):
    """Coordinates standard epoch training iterations across an Executor loop."""

    executor: StepExecutor[Any]
    epochs: int = 200
    seed: int = SEED
    description: str = "Supervised Workflow"

    @override
    def run(self) -> Tensor:
        set_seed(self.seed)
        final_loss = Tensor(0.0)
        for _epoch in track(
            range(self.epochs), description=f"[bold]{self.description}[/]"
        ):
            final_loss = self.executor.step_epoch()
        return final_loss


@dataclass(kw_only=True)
class SupervisedStepExecutor(Generic[PolicyT, RawDataT, BatchT], StepExecutor[PolicyT]):
    """Assembles structural execution blocks to process a single epoch pass."""

    streamer: Streamer[RawDataT]
    collator: Collator[RawDataT, BatchT]
    evaluator: Evaluator[PolicyT, BatchT, Any]
    criterion: Criterion[Any, BatchT]
    hooks: Sequence[Hook[PolicyT]] = field(default_factory=list[Hook[PolicyT]])

    @override
    def step_epoch(self) -> Tensor:
        self.policy.train()
        self.optimiser.zero_grad()

        epoch_loss = torch.zeros((), device=self.device_)
        epoch_denom = torch.zeros((), device=self.device_)

        for raw_data in self.streamer:
            batch = self.collator(raw_data, self.device_)
            predictions = self.evaluator.execute(self.policy, batch)
            loss_val, denominator = self.criterion(predictions, batch)

            epoch_loss += loss_val
            epoch_denom += denominator

        final_loss = epoch_loss / (epoch_denom + EPS)
        final_loss.backward()  # type: ignore[no-untyped-call]  # pyright: ignore[reportUnknownMemberType]

        for hook in self.hooks:
            hook.apply(self.policy, self.optimiser)

        self.optimiser.step()
        return final_loss.detach()


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RawTrajectory:
    states: Any
    targets: Any
    weights: Any | None = None


@dataclass
class RawTrajectoryStreamer(
    Generic[NumPoints, DimState, DimAction], Streamer[RawTrajectory]
):
    """Extracts raw trajectories sequentially."""

    states: States[NumPoints, DimState]
    targets: Actions[NumPoints, DimAction]
    weights: Vector[NumPoints] | None = None

    @property
    def length(self) -> NumPoints:
        assert self.states.length == self.targets.length
        if self.weights is not None:
            assert self.states.length == len(self.weights)
        return self.states.length

    @override
    def __iter__(self) -> Iterator[RawTrajectory]:
        for i in range(self.length):
            yield RawTrajectory(
                states=self.states[i],
                targets=self.targets[i],
                weights=self.weights[i] if self.weights is not None else None,
            )


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ImitationBatch:
    states: Tensor  # (batch, state_dim)
    targets: Tensor  # (batch, action_dim)
    weights: Tensor  # (batch,)


class ImitationBatchCollator(Collator[RawTrajectory, ImitationBatch]):
    """Processes extracted raw trajectory bundles directly into target device tensors."""

    @override
    def __call__(self, raw_data: RawTrajectory, device: torch.device) -> ImitationBatch:
        states = torch.tensor(raw_data.states, dtype=torchDType, device=device)
        targets = torch.tensor(raw_data.targets, dtype=torchDType, device=device)

        if raw_data.weights is not None:
            weights = torch.tensor(raw_data.weights, dtype=torchDType, device=device)
        else:
            weights = torch.ones(states.size(0), dtype=torchDType, device=device)

        return ImitationBatch(states=states, targets=targets, weights=weights)


# ──────────────────────────────────────────────────────────────────────────────


class ForwardPassEvaluator(
    Generic[PolicyT, PredictionT], Evaluator[PolicyT, ImitationBatch, PredictionT]
):
    """Passes states directly to policy."""

    @override
    def execute(self, policy: PolicyT, batch: ImitationBatch) -> PredictionT:
        return cast(PredictionT, policy(batch.states))


# ── Hooks ─────────────────────────────────────────────────────────────────────


@dataclass
class GradientClipper(Hook[PolicyT]):
    max_norm: float = 1.0

    @override
    def apply(self, policy: PolicyT, optimiser: torch.optim.Optimizer) -> None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_norm)


@dataclass
class WeightDecayScheduler(Hook[PolicyT]):
    factor: float = 0.99

    @override
    def apply(self, policy: PolicyT, optimiser: torch.optim.Optimizer) -> None:
        for group in optimiser.param_groups:
            if "weight_decay" in group:
                group["weight_decay"] *= self.factor


# ── Losses ────────────────────────────────────────────────────────────────────


class WeightedHuberLoss(Criterion[Tensor, ImitationBatch]):
    """Computes a sample-weighted Huber loss function."""

    @override
    def __call__(
        self, predictions: Tensor, batch: ImitationBatch
    ) -> tuple[Tensor, Tensor]:
        huber = F.huber_loss(predictions, batch.targets, reduction="none")
        huber_per_sample = huber.mean(dim=-1) * batch.weights
        return huber_per_sample.sum(), batch.weights.sum()


class GaussianMixtureNLLLoss(Criterion[GMMOutputs, ImitationBatch]):
    """Computes sample-weighted multi-modal Gaussian Mixture Negative Log-Likelihood."""

    @override
    def __call__(
        self, predictions: GMMOutputs, batch: ImitationBatch
    ) -> tuple[Tensor, Tensor]:
        targets_exp = batch.targets.unsqueeze(1)  # (batch, 1, action_dim)

        variance = predictions.sigma**2
        exponent = -0.5 * ((targets_exp - predictions.mu) ** 2) / (variance + EPS)
        normaliser = torch.sqrt(2.0 * torch.pi * variance + EPS)

        prob_per_dim = torch.exp(exponent) / normaliser
        prob_components = torch.prod(prob_per_dim, dim=-1)  # (batch, n_components)
        prob_mixture = torch.sum(predictions.pi * prob_components, dim=-1)  # (batch,)

        nll = -torch.log(prob_mixture + EPS)
        weighted_nll = nll * batch.weights

        return weighted_nll.sum(), batch.weights.sum()


# ── Model Policies ────────────────────────────────────────────────────────────


class BCPolicy(nn.Module, RuntimeGeneric[DimState, DimAction]):
    """Pure neural network policy mapping states to actions."""

    def __init__(
        self, state_dim: DimState, action_dim: DimAction, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.network: nn.Module = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    @override
    def forward(self, states: Tensor) -> Tensor:
        return cast(Tensor, self.network(states))


class GMMPolicy(nn.Module, RuntimeGeneric[DimState, DimAction]):
    """
    Gaussian Mixture Model policy that maps states to a multi-modal action
    distribution parameterised by mixing coefficients, means, and variances.
    """

    def __init__(
        self,
        state_dim: DimState,
        action_dim: DimAction,
        hidden_dim: int = 128,
        n_components: int = 5,
    ) -> None:
        super().__init__()
        self.state_dim: DimState = state_dim
        self.action_dim: DimAction = action_dim
        self.n_components: int = n_components

        self.backbone: nn.Module = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi_head: nn.Module = nn.Linear(hidden_dim, n_components)
        self.mu_head: nn.Module = nn.Linear(hidden_dim, n_components * action_dim)
        self.sigma_head: nn.Module = nn.Linear(hidden_dim, n_components * action_dim)

    @override
    def forward(
        self,
        states: Tensor,  # (batch, state_dim)
    ) -> GMMOutputs:
        features = self.backbone(states)
        batch_size = states.size(0)

        pi = torch.softmax(self.pi_head(features), dim=-1)  # (batch, n_components)
        mu = Tensor(self.mu_head(features)).view(
            batch_size, self.n_components, self.action_dim
        )  # (batch, n_components, action_dim)
        sigma = torch.exp(self.sigma_head(features)).view(
            batch_size, self.n_components, self.action_dim
        )  # (batch, n_components, action_dim)

        return GMMOutputs(pi=pi, mu=mu, sigma=sigma)


## ─────────────────────────────────────────────────────────────────────────────
