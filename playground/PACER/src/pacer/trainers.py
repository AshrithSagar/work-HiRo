"""
Policy training
=======
"""
# src/pacer/trainers.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Generic, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from typingkit.numpy import enforce_shapes

from pacer.base import Demonstrations
from pacer.pacer import PACER
from pacer.phase import PhaseEstimator
from pacer.typings import (
    Action,
    Actions,
    ActionsCollection,
    DimAction,
    DimState,
    NumBins,
    NumDemos,
    NumPoints,
    States,
    TrustValuesCollection,
    npDType,
    torchDType,
)
from pacer.utils import SEED, get_torch_device_auto, set_seed

## ── Policies ─────────────────────────────────────────────────────────────────


class BCPolicy(nn.Module, Generic[DimState, DimAction]):
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


@dataclass
class BCTrainer(Generic[NumDemos, NumPoints, DimState, DimAction]):
    """Behavioral cloning policy trainer."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    device: torch.device = field(kw_only=True, default_factory=get_torch_device_auto)
    seed: int = SEED
    ##
    policy: BCPolicy[DimState, DimAction] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def compute_huber_loss(self) -> Tensor:  # L
        loss = torch.tensor(0.0, dtype=torchDType, device=self.device)
        total_samples: int = 0
        for demo in self.demonstrations:
            states = torch.tensor(
                np.array(demo.states), dtype=torchDType, device=self.device
            )  # (T_i, state_dim)
            targets = torch.tensor(
                np.array(demo.actions), dtype=torchDType, device=self.device
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
        self.policy = policy.to(self.device)
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

    @enforce_shapes
    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(np.array(states)).float().to(self.device)
            actions_tensor: Tensor = self.policy(states_tensor)
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            Action[DimAction](action_np) for action_np in actions_np
        )
        return actions


@dataclass
class PACERBCTrainer(Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    """PACER + Behavioral cloning policy trainer."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    device: torch.device = field(kw_only=True, default_factory=get_torch_device_auto)
    seed: int = SEED
    ##
    phase_estimator: PhaseEstimator[NumDemos, NumPoints, DimState, DimAction] = field(
        init=False
    )
    pacer: PACER[NumBins, NumDemos, NumPoints, DimState, DimAction] = field(init=False)
    trust_values: TrustValuesCollection[NumDemos, NumPoints] = field(init=False)
    pseudo_labels: ActionsCollection[NumDemos, NumPoints, DimAction] = field(init=False)
    policy: BCPolicy[DimState, DimAction] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def prepare(
        self,
        *,
        phase_hidden_dim: int = 128,
        phase_margin: float = 1.0,
        phase_lr: float = 1e-3,
        phase_epochs: int = 240,
        n_bins: NumBins = cast(NumBins, 96),
        tukey_cutoff: npDType | float = 4.685,  # c
        min_trust: npDType | float = 0.02,  # w_min
        debias_weight: npDType | float = 0.5,  # lambda_{debias}
        sideways_attenuation_shrinkage: npDType | float = 0.5,  # rho_0
        speed_regularisation_influence: npDType | float = 0.5,  # eta_0
        temporal_smoothing_weight: npDType | float = 0.0,  # kappa
    ) -> Tensor:
        set_seed(self.seed)
        self.phase_estimator = PhaseEstimator(self.demonstrations, device=self.device)
        loss = self.phase_estimator.train(
            hidden_dim=phase_hidden_dim,
            margin=phase_margin,
            lr=phase_lr,
            epochs=phase_epochs,
        )
        self.pacer = PACER(self.phase_estimator, n_bins=n_bins)
        self.pacer.make_bins()
        self.trust_values = self.pacer.compute_trust_values(
            cutoff=tukey_cutoff,
            min_trust=min_trust,
        )
        self.pseudo_labels = self.pacer.compute_pseudo_labels(
            self.trust_values,
            debias_weight=debias_weight,
            sideways_attenuation_shrinkage=sideways_attenuation_shrinkage,
            speed_regularisation_influence=speed_regularisation_influence,
            temporal_smoothing_weight=temporal_smoothing_weight,
        )
        return loss

    def compute_huber_loss(self) -> Tensor:  # L
        loss = torch.tensor(0.0, dtype=torchDType, device=self.device)
        total_weight = torch.tensor(0.0, dtype=torchDType, device=self.device)
        for i, demo in enumerate(self.demonstrations):
            states = torch.tensor(
                np.array(demo.states), dtype=torchDType, device=self.device
            )  # (T_i, state_dim)
            targets = torch.tensor(
                np.array(self.pseudo_labels[i]), dtype=torchDType, device=self.device
            )  # (T_i, action_dim)
            weights = torch.tensor(
                np.array(self.trust_values[i]), dtype=torchDType, device=self.device
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
        self.policy = policy.to(self.device)
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

    @enforce_shapes
    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(np.array(states)).float().to(self.device)
            actions_tensor: Tensor = self.policy(states_tensor)
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            [Action[DimAction](action_np) for action_np in actions_np]
        )
        return actions


## ─────────────────────────────────────────────────────────────────────────────
