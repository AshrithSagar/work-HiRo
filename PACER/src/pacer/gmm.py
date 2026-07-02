"""
GMM
=======
"""
# src/pacer/gmm.py

# pyright: reportPrivateImportUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import override

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track
from torch import Tensor
from torch._prims_common import DeviceLikeType
from typingkit.core import RuntimeGeneric, TypedList, TypedMapping

from pacer.base import Action, Actions, ActionsCollection, States, StatesCollection
from pacer.typings import (
    DemoIndex,
    DimAction,
    DimState,
    NumDemos,
    NumPoints,
    npDType,
    torchDType,
)
from pacer.utils import SEED, TORCH_DEVICE, get_torch_device, set_seed

## ── Policy ───────────────────────────────────────────────────────────────────


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
    ) -> tuple[Tensor, Tensor, Tensor]:  # (pi, mu, sigma)
        features = self.backbone(states)
        batch_size = states.size(0)

        pi = torch.softmax(self.pi_head(features), dim=-1)  # (batch, n_components)
        mu = Tensor(self.mu_head(features)).view(
            batch_size, self.n_components, self.action_dim
        )  # (batch, n_components, action_dim)
        sigma = torch.exp(self.sigma_head(features)).view(
            batch_size, self.n_components, self.action_dim
        )  # (batch, n_components, action_dim)

        return pi, mu, sigma


## ── Trainers ─────────────────────────────────────────────────────────────────


@dataclass
class GMMTrainConfig:
    hidden_dim: int = 128
    n_components: int = 5
    lr: float = 1e-3
    epochs: int = 240


@dataclass
class WeightedGMMTrainer(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    """Gaussian Mixture Model policy trainer supporting sequence sample weights."""

    states: StatesCollection[NumDemos, NumPoints, DimState]
    targets: ActionsCollection[NumDemos, NumPoints, DimAction]
    weights: TypedMapping[NumDemos, DemoIndex, TypedList[NumPoints, npDType]]
    _: KW_ONLY
    device: InitVar[DeviceLikeType] = TORCH_DEVICE
    seed: int = SEED
    ##
    n_demos: NumDemos = field(init=False)
    device_: torch.device = field(init=False)
    policy: GMMPolicy[DimState, DimAction] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)
        assert self.states.length == self.targets.length == self.weights.length
        self.n_demos = self.states.length

    def compute_nll_loss(self) -> Tensor:
        loss = torch.tensor(0.0, dtype=torchDType, device=self.device_)
        total_weight = torch.tensor(0.0, dtype=torchDType, device=self.device_)

        for i in range(self.n_demos):
            states = torch.tensor(
                self.states[i].numpy(), dtype=torchDType, device=self.device_
            )  # (T_i, state_dim)
            targets = torch.tensor(
                self.targets[i].numpy(), dtype=torchDType, device=self.device_
            )  # (T_i, action_dim)
            weights = torch.tensor(
                np.array(self.weights[i]),
                dtype=torchDType,
                device=self.device_,
            )  # (T_i,)

            pi, mu, sigma = self.policy(states)
            targets_exp = targets.unsqueeze(1)  # (T_i, 1, action_dim)
            variance = sigma**2
            exponent = -0.5 * ((targets_exp - mu) ** 2) / (variance + 1e-6)

            normaliser = torch.sqrt(2.0 * np.pi * variance + 1e-6)
            prob_per_dim = torch.exp(exponent) / normaliser
            prob_components = torch.prod(prob_per_dim, dim=-1)  # (T_i, n_components)
            prob_mixture = torch.sum(pi * prob_components, dim=-1)  # (T_i,)
            nll = -torch.log(prob_mixture + 1e-6)  # (T_i,)

            weighted_nll = nll * weights  # (T_i,)
            loss += weighted_nll.sum()
            total_weight += weights.sum()

        if total_weight.item() > 0:
            loss /= total_weight
        return loss

    def train(self, config: GMMTrainConfig) -> Tensor:
        """Train GMM policy using maximum likelihood estimation (NLL loss)."""
        set_seed(self.seed)
        policy = GMMPolicy(
            state_dim=self.states.dim,
            action_dim=self.targets.dim,
            hidden_dim=config.hidden_dim,
            n_components=config.n_components,
        )
        self.policy = policy.to(self.device_)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        self.policy.train()
        loss = self.compute_nll_loss()
        for _epoch in track(
            range(config.epochs), description="[bold]GMM Policy training[/]"
        ):
            self.optimiser.zero_grad()
            loss = self.compute_nll_loss()
            loss.backward()  # type: ignore[no-untyped-call]  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        """Predicts actions by taking the mean of the most probable mixture component."""
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(states.numpy()).float().to(self.device_)
            pi, mu, _ = self.policy(states_tensor)
            # Component index with the highest mixing weight (pi)
            best_component = torch.argmax(pi, dim=-1)  # (T,)
            batch_idx = torch.arange(states_tensor.size(0), device=self.device_)
            actions_tensor: Tensor = mu[batch_idx, best_component]  # (T, action_dim)
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            [Action[DimAction](action_np) for action_np in actions_np]
        )
        return actions


## ─────────────────────────────────────────────────────────────────────────────
