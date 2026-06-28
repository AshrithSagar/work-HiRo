"""
Imitation Learning Base
=======
"""
# src/pacer/imitation/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Self, cast, override

import torch
import torch.nn as nn
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
from pacer.pacer.pseudolabel import PseudoLabels
from pacer.pacer.trust.base import TrustValuesCollection
from pacer.phase import PhasesCollection
from pacer.typings import DimAction, DimState, NumDemos, NumPoints, torchDType
from pacer.utils import SEED, TORCH_DEVICE, get_torch_device, set_seed

## ── Imitation Learning ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ILDataset(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    """Dataset consumed by imitation learning algorithms."""

    states: StatesCollection[NumDemos, NumPoints, DimState]
    actions: ActionsCollection[NumDemos, NumPoints, DimAction]
    weights: TrustValuesCollection[NumDemos, NumPoints] | None = None
    phases: PhasesCollection[NumDemos, NumPoints] | None = None

    def __post_init__(self) -> None:
        assert self.states.length == self.actions.length
        if self.weights is not None:
            assert self.weights.length == self.states.length
        if self.phases is not None:
            assert self.phases.length == self.states.length

    @classmethod
    def from_demonstrations(
        cls,
        demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction],
        *,
        weights: TrustValuesCollection[NumDemos, NumPoints] | None = None,
        phases: PhasesCollection[NumDemos, NumPoints] | None = None,
    ) -> Self:
        """Construct an imitation dataset directly from demonstrations."""
        return cls(
            states=demonstrations.states,
            actions=demonstrations.actions,
            weights=weights,
            phases=phases,
        )

    @classmethod
    def from_pseudo_labels(
        cls,
        demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction],
        pseudo_labels: PseudoLabels[NumDemos, NumPoints, DimState, DimAction],
        *,
        weights: TrustValuesCollection[NumDemos, NumPoints] | None = None,
        phases: PhasesCollection[NumDemos, NumPoints] | None = None,
    ) -> Self:
        """
        Construct a dataset using PACER pseudo-labels.\\
        Falls back to the original demonstration states if PACER did not generate refined states.
        """
        return cls(
            states=pseudo_labels.states or demonstrations.states,
            actions=pseudo_labels.actions,
            weights=weights,
            phases=phases,
        )


@dataclass(slots=True)
class ILTrainConfig:
    """Base training configuration shared by all imitation learners."""

    lr: float = 1e-3
    epochs: int = 200
    gradient_clip_norm: float | None = 1.0


class ILPolicy(nn.Module, RuntimeGeneric[DimState, DimAction], ABC):
    """Base class of all imitation policies."""

    @override
    @abstractmethod
    def forward(self, states: Tensor) -> Tensor:
        """
        states -> Shape depends on the concrete learner.\\
        returns -> Predicted actions.
        """
        raise NotImplementedError


@dataclass(slots=True)
class ILObjective(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction], ABC):
    """Computes the optimisation objective for an imitation learner."""

    @abstractmethod
    def __call__(
        self,
        *,
        policy: ILPolicy[DimState, DimAction],
        dataset: ILDataset[NumDemos, NumPoints, DimState, DimAction],
        device: torch.device,
    ) -> Tensor: ...


@dataclass
class ILTrainer(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction], ABC):
    """Generic imitation learner."""

    dataset: ILDataset[NumDemos, NumPoints, DimState, DimAction]
    policy: ILPolicy[DimState, DimAction]
    objective: ILObjective[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    device: InitVar[DeviceLikeType] = TORCH_DEVICE
    seed: int = SEED
    ##
    device_: torch.device = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)

    def train(self, config: ILTrainConfig) -> Tensor:
        set_seed(self.seed)
        self.policy.to(self.device_)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=config.lr)

        self.policy.train()
        loss = self.objective(
            policy=self.policy, dataset=self.dataset, device=self.device_
        )
        for _epoch in track(
            range(config.epochs), description="[bold]Policy training[/]"
        ):
            self.optimiser.zero_grad()
            loss = self.objective(
                policy=self.policy, dataset=self.dataset, device=self.device_
            )
            loss.backward()  # type: ignore[no-untyped-call]  # pyright: ignore[reportUnknownMemberType]
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.policy.parameters(),
                    max_norm=config.gradient_clip_norm,
                )
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    def predict(
        self, states: States[NumPoints, DimState]
    ) -> Actions[NumPoints, DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = torch.as_tensor(
                states.numpy(), dtype=torchDType, device=self.device_
            )
            actions_tensor = cast(Tensor, self.policy(states_tensor))
            actions_np = actions_tensor.cpu().numpy()
        actions = Actions[NumPoints, DimAction](
            [Action[DimAction](action_np) for action_np in actions_np]
        )
        return actions


## ─────────────────────────────────────────────────────────────────────────────
