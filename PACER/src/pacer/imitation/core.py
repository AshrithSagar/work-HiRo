"""
Core
=======
"""
# src/pacer/imitation/core.py

## ── Imports ──────────────────────────────────────────────────────────────────

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from torch._prims_common import DeviceLikeType

from pacer.utils import SEED, TORCH_DEVICE, get_torch_device

## ── Typings ──────────────────────────────────────────────────────────────────

RawDataT = TypeVar("RawDataT")
PolicyT = TypeVar("PolicyT", bound=nn.Module)
BatchT = TypeVar("BatchT")
PredictionT = TypeVar("PredictionT")

## ── Imitation Learning ───────────────────────────────────────────────────────


class Workflow(ABC):
    """Outer control loop that orchestrates the lifecycle of an experiment."""

    @abstractmethod
    def run(self) -> Any: ...


@dataclass(kw_only=True)
class StepExecutor(Generic[PolicyT], ABC):
    """Manages tracking hardware state, model policy parameters, and execution hooks."""

    policy: PolicyT
    device: InitVar[DeviceLikeType] = TORCH_DEVICE
    seed: int = SEED
    lr: float = 1e-3
    ##
    device_: torch.device = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def __post_init__(self, device: DeviceLikeType) -> None:
        self.device_ = get_torch_device(device)
        self.policy.to(self.device_)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    @abstractmethod
    def step_epoch(self) -> Tensor:
        """Executes a complete parameter update across the streamed collection."""
        ...


class Streamer(Generic[RawDataT], ABC):
    """Sequentially extracts trajectories from raw datasets."""

    @abstractmethod
    def __iter__(self) -> Iterator[RawDataT]: ...


class Collator(Generic[RawDataT, BatchT], ABC):
    """Converts raw items or sequence groups into processed tensor batches."""

    @abstractmethod
    def __call__(self, raw_data: RawDataT, device: torch.device) -> BatchT: ...


class Evaluator(Generic[PolicyT, BatchT, PredictionT], ABC):
    """Passes raw or processed batch elements into the policy network."""

    @abstractmethod
    def execute(self, policy: PolicyT, batch: BatchT) -> PredictionT: ...


class Criterion(Generic[PredictionT, BatchT], ABC):
    """Evaluates mathematical loss/error calculations."""

    @abstractmethod
    def __call__(
        self, predictions: PredictionT, batch: BatchT
    ) -> tuple[Tensor, Tensor]:
        """Returns: (total_unnormalised_loss, total_denominator)."""
        ...


class Hook(Generic[PolicyT], ABC):
    """Intercepts the optimisation pass after `backward()` to modify gradients or weights."""

    @abstractmethod
    def apply(self, policy: PolicyT, optimiser: torch.optim.Optimizer) -> None: ...


## ─────────────────────────────────────────────────────────────────────────────
