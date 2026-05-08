"""
Policy training
=======
"""
# src/pacer/trainers.py

# pyright: reportPrivateImportUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import cast, override

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from torch._prims_common import DeviceLikeType
from typingkit.core import RuntimeGeneric, TypedList, TypedMapping

from pacer.base import (
    Action,
    Actions,
    ActionsCollection,
    Demonstrations,
    States,
    StatesCollection,
)
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

## ── Policies ─────────────────────────────────────────────────────────────────


class BCPolicy(nn.Module, RuntimeGeneric[DimState, DimAction]):
    """Behavioral cloning policy that maps states to actions."""

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
    def forward(
        self,
        states: Tensor,  # (batch, state_dim)
    ) -> Tensor:  # (batch,)
        return cast(Tensor, self.network(states))


## ── Trainers ─────────────────────────────────────────────────────────────────


@dataclass
class BCTrainer(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    """Behavioral cloning policy trainer."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    device: InitVar[DeviceLikeType] = TORCH_DEVICE
    seed: int = SEED
    ##
    device_: torch.device = field(init=False)
    policy: BCPolicy[DimState, DimAction] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)

    def compute_huber_loss(self) -> Tensor:  # L
        loss = torch.tensor(0.0, dtype=torchDType, device=self.device_)
        total_samples: int = 0
        for demo in self.demonstrations:
            states = torch.tensor(
                demo.states.numpy(), dtype=torchDType, device=self.device_
            )  # (T_i, state_dim)
            targets = torch.tensor(
                demo.actions.numpy(), dtype=torchDType, device=self.device_
            )  # (T_i, action_dim)
            preds = cast(Tensor, self.policy(states))  # (T_i, action_dim)

            diffs: Tensor = preds - targets  # (T_i, action_dim)
            demo_loss = F.huber_loss(
                diffs, torch.zeros_like(diffs), reduction="sum"
            )  # (T_i, action_dim)
            loss += demo_loss
            total_samples += demo.length  # T_i
        if total_samples > 0:
            loss /= total_samples  # Normalise over samples
        return loss

    def train(
        self,
        *,
        policy_hidden_dim: int = 128,
        policy_lr: float = 1e-3,
        policy_epochs: int = 240,
    ) -> Tensor:
        """Train BC policy using weighted Huber loss."""
        set_seed(self.seed)
        policy = BCPolicy(
            state_dim=self.demonstrations.state_dim,
            action_dim=self.demonstrations.action_dim,
            hidden_dim=policy_hidden_dim,
        )
        self.policy = policy.to(self.device_)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.policy.train()
        loss = self.compute_huber_loss()
        for _epoch in track(
            range(policy_epochs), description="[bold]Policy training[/]"
        ):
            self.optimiser.zero_grad()
            loss = self.compute_huber_loss()
            loss.backward()  # type: ignore[no-untyped-call]  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(states.numpy()).float().to(self.device_)
            actions_tensor = cast(Tensor, self.policy(states_tensor))
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            [Action[DimAction](action_np) for action_np in actions_np]
        )
        return actions


@dataclass
class WeightedBCTrainer(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    """Weighted behavioral cloning policy trainer."""

    states: StatesCollection[NumDemos, NumPoints, DimState]
    targets: ActionsCollection[NumDemos, NumPoints, DimAction]
    weights: TypedMapping[NumDemos, DemoIndex, TypedList[NumPoints, npDType]]
    _: KW_ONLY
    device: InitVar[DeviceLikeType] = TORCH_DEVICE
    seed: int = SEED
    ##
    n_demos: NumDemos = field(init=False)
    device_: torch.device = field(init=False)
    policy: BCPolicy[DimState, DimAction] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)
        assert self.states.length == self.targets.length == self.weights.length
        self.n_demos = self.states.length

    def compute_huber_loss(self) -> Tensor:  # L
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
            preds = cast(Tensor, self.policy(states))  # (T_i, action_dim)

            diffs: Tensor = preds - targets  # (T_i, action_dim)
            huber_losses = F.huber_loss(
                diffs, torch.zeros_like(diffs), reduction="none"
            )  # (T_i, action_dim)
            huber_losses = huber_losses.mean(dim=1)  # (T_i,)
            weighted_losses = huber_losses * weights  # (T_i,)

            loss += weighted_losses.sum()
            total_weight += weights.sum()

        if total_weight.item() > 0:
            loss /= total_weight
        return loss

    def train(
        self,
        *,
        policy_hidden_dim: int = 128,
        policy_lr: float = 1e-3,
        policy_epochs: int = 240,
    ) -> Tensor:
        """Train BC policy using weighted Huber loss."""
        set_seed(self.seed)
        policy = BCPolicy(
            state_dim=self.states.dim,
            action_dim=self.targets.dim,
            hidden_dim=policy_hidden_dim,
        )
        self.policy = policy.to(self.device_)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.policy.train()
        loss = self.compute_huber_loss()
        for _epoch in track(
            range(policy_epochs), description="[bold]Policy training[/]"
        ):
            self.optimiser.zero_grad()
            loss = self.compute_huber_loss()
            loss.backward()  # type: ignore[no-untyped-call]  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(states.numpy()).float().to(self.device_)
            actions_tensor = cast(Tensor, self.policy(states_tensor))
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            [Action[DimAction](action_np) for action_np in actions_np]
        )
        return actions


## ─────────────────────────────────────────────────────────────────────────────
