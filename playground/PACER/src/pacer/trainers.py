"""
Policy training
=======
"""
# src/pacer/trainers.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import InitVar, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from torch._prims_common import DeviceLikeType
from typingkit.core import RuntimeGeneric

from pacer.base import (
    Action,
    Actions,
    ActionsCollection,
    Demonstrations,
    States,
    StatesCollection,
)
from pacer.pacer import Bins, TrustValuesCollection
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints, torchDType
from pacer.utils import SEED, TORCH_DEVICE, get_torch_device, set_seed

## ── Policies ─────────────────────────────────────────────────────────────────


class BCPolicy(nn.Module, RuntimeGeneric[DimState, DimAction]):
    """Behavioral cloning policy that maps states to actions."""

    def __init__(
        self, state_dim: DimState, action_dim: DimAction, hidden_dim: int = 128
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        states: Tensor,  # (batch, state_dim)
    ) -> Tensor:  # (batch,)
        return self.network(states)  # type: ignore[no-any-return]  # ty: ignore[unused-ignore-comment]


## ── Trainers ─────────────────────────────────────────────────────────────────


@dataclass
class BCTrainer(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    """Behavioral cloning policy trainer."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    device: InitVar[DeviceLikeType] = field(default=TORCH_DEVICE, kw_only=True)
    seed: int = field(default=SEED, kw_only=True)
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
            preds: Tensor = self.policy(states)  # (T_i, action_dim)

            diffs: Tensor = preds - targets  # (T_i, action_dim)
            demo_loss = F.huber_loss(
                diffs, torch.zeros_like(diffs), reduction="sum"
            )  # (T_i, action_dim)
            loss += demo_loss
            total_samples += demo.__len__()  # T_i
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
            loss.backward()  # type: ignore[no-untyped-call]  # ty: ignore[unused-ignore-comment]
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(states.numpy()).float().to(self.device_)
            actions_tensor: Tensor = self.policy(states_tensor)
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            Action[DimAction](action_np) for action_np in actions_np
        )
        return actions


@dataclass
class PACERBCTrainer(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    """PACER + Behavioral cloning policy trainer."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    trust_values: TrustValuesCollection[NumDemos, NumPoints]
    pseudo_labels: ActionsCollection[NumDemos, NumPoints, DimAction]
    state_labels: StatesCollection[NumDemos, NumPoints, DimState] | None = None
    device: InitVar[DeviceLikeType] = field(default=TORCH_DEVICE, kw_only=True)
    seed: int = field(default=SEED, kw_only=True)
    ##
    device_: torch.device = field(init=False)
    policy: BCPolicy[DimState, DimAction] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)

    def compute_huber_loss(self) -> Tensor:  # L
        loss = torch.tensor(0.0, dtype=torchDType, device=self.device_)
        total_weight = torch.tensor(0.0, dtype=torchDType, device=self.device_)
        for i, demo in enumerate(self.demonstrations):
            if self.state_labels is not None:
                states = torch.tensor(
                    self.state_labels[i].numpy(), dtype=torchDType, device=self.device_
                )  # (T_i, state_dim)
            else:
                states = torch.tensor(
                    demo.states.numpy(), dtype=torchDType, device=self.device_
                )  # (T_i, state_dim)
            targets = torch.tensor(
                self.pseudo_labels[i].numpy(),
                dtype=torchDType,
                device=self.device_,
            )  # (T_i, action_dim)
            weights = torch.tensor(
                np.array(self.trust_values[i]),
                dtype=torchDType,
                device=self.device_,
            )  # (T_i,)
            preds: Tensor = self.policy(states)  # (T_i, action_dim)

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
        """Train PACER policy using weighted Huber loss with pseudo-labels."""
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
            loss.backward()  # type: ignore[no-untyped-call]  # ty: ignore[unused-ignore-comment]
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(states.numpy()).float().to(self.device_)
            actions_tensor: Tensor = self.policy(states_tensor)
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            [Action[DimAction](action_np) for action_np in actions_np]
        )
        return actions


## ─────────────────────────────────────────────────────────────────────────────
