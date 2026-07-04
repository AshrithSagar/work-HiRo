"""
Experiments
=======
"""
# src/pacer/imitation/experiments.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Any, Generic, Self, override

import torch
from torch import Tensor

from pacer.base import Demonstrations, StatesCollection
from pacer.imitation.core import Collator, Criterion, Evaluator, Hook, Streamer
from pacer.imitation.supervised import (
    BCPolicy,
    ForwardPassEvaluator,
    GradientClipper,
    ImitationBatchCollator,
    RawTrajectory,
    RawTrajectoryStreamer,
    SupervisedStepExecutor,
    SupervisedWorkflow,
    WeightedHuberCriterion,
)
from pacer.pacer import PACER, PACERConfig, PACERResult
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints

## ── Experiments ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExperimentResult(Generic[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    policy: Any
    loss: Tensor


@dataclass(frozen=True)
class PACERExperimentResult(
    Generic[NumBins, NumDemos, NumPoints, DimState, DimAction],
    ExperimentResult[NumDemos, NumPoints, DimState, DimAction],
):
    pacer_result: PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 240
    hidden_dim: int = 128
    lr: float = 1e-3
    max_norm: float = 1.0


@dataclass
class ImitationExperiment(Generic[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    config: TrainingConfig = field(default_factory=TrainingConfig)
    ##
    _policy: torch.nn.Module = field(init=False)
    _streamer: Streamer[Any] = field(init=False)
    _collator: Collator[Any, Any] = field(init=False)
    _evaluator: Evaluator[Any, Any, Any] = field(init=False)
    _criterion: Criterion[Any, Any] = field(init=False)
    _hooks: list[Hook[Any]] = field(init=False)

    def __post_init__(self) -> None:
        self._policy = self.policy()
        self._streamer = self.streamer()
        self._collator = self.collator()
        self._evaluator = self.evaluator()
        self._criterion = self.criterion()
        self._hooks = self.hooks()

    # ──────────────────────────────────────────────────────────────────────────

    def policy(self) -> torch.nn.Module:
        return BCPolicy(
            self.demonstrations.states.dim,
            self.demonstrations.actions.dim,
            self.config.hidden_dim,
        )

    def streamer(self) -> Streamer[RawTrajectory]:
        return RawTrajectoryStreamer(
            self.demonstrations.states, self.demonstrations.actions
        )

    def collator(self) -> Collator[Any, Any]:
        return ImitationBatchCollator()

    def evaluator(self) -> Evaluator[Any, Any, Any]:
        return ForwardPassEvaluator()

    def criterion(self) -> Criterion[Any, Any]:
        return WeightedHuberCriterion()

    def hooks(self) -> list[Hook[Any]]:
        return [GradientClipper(max_norm=self.config.max_norm)]

    # ──────────────────────────────────────────────────────────────────────────

    def with_policy(self, policy: torch.nn.Module) -> Self:
        self._policy = policy
        return self

    def with_streamer(self, streamer: Streamer[Any]) -> Self:
        self._streamer = streamer
        return self

    def with_collator(self, collator: Collator[Any, Any]) -> Self:
        self._collator = collator
        return self

    def with_criterion(self, criterion: Criterion[Any, Any]) -> Self:
        self._criterion = criterion
        return self

    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> ExperimentResult[NumDemos, NumPoints, DimState, DimAction]:
        """Assembles the current pipeline blocks and runs the optimiser loop."""
        loss = SupervisedWorkflow(
            executor=SupervisedStepExecutor(
                policy=self._policy,
                streamer=self._streamer,
                collator=self._collator,
                evaluator=self._evaluator,
                criterion=self._criterion,
                hooks=self._hooks,
                lr=self.config.lr,
            ),
            epochs=self.config.epochs,
            description="Imitation Run",
        ).run()
        return ExperimentResult(
            demonstrations=self.demonstrations, policy=self._policy, loss=loss
        )


class PACERImitationExperiment(
    Generic[NumBins, NumDemos, NumPoints, DimState, DimAction],
    ImitationExperiment[NumDemos, NumPoints, DimState, DimAction],
):
    def __init__(
        self,
        demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction],
        pacer_config: PACERConfig[NumBins],
        config: TrainingConfig | None = None,
    ) -> None:
        self.pacer_result: PACERResult[
            NumBins, NumDemos, NumPoints, DimState, DimAction
        ] = PACER(demonstrations, config=pacer_config).run()
        self.target_states: StatesCollection[NumDemos, NumPoints, DimState] = (
            self.pacer_result.pseudo_labels.states
            if self.pacer_result.pseudo_labels.states is not None
            else demonstrations.states
        )
        super().__init__(demonstrations, config or TrainingConfig())

    @override
    def streamer(self) -> Streamer[Any]:
        return RawTrajectoryStreamer(
            states=self.target_states,
            targets=self.pacer_result.pseudo_labels.actions,
            weights=self.pacer_result.action_trust_values,
        )


## ─────────────────────────────────────────────────────────────────────────────
