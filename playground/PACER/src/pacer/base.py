"""
PACER Base
=======
Implementation follows the following paper from
Shreyas Kumar & Ravi Prakash, CoRL 2025 Workshop on Robot Data:
"PACER: Progress-Aligned Curation for Error-Resilient Imitation Learning"
https://openreview.net/forum?id=gaYyBvP2Rz
"""
# src/pacer/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, NamedTuple, TypeAlias, TypeVar, cast, overload

import numpy as np
import numpy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from typingkit.core import TypedList
from typingkit.numpy import enforce_shapes
from typingkit.numpy._typed.helpers import Array1D

from pacer.utils import (
    EPS,
    MAD_SCALE,
    SEED,
    get_torch_device_auto,
    median,
    normalise,
    npDType,
    set_seed,
    torchDType,
)

set_seed(SEED)

## ── Typings ──────────────────────────────────────────────────────────────────

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
NumDemos = TypeVar("NumDemos", bound=int, default=int)  # N
NumBins = TypeVar("NumBins", bound=int, default=int)  # B

State: TypeAlias = Array1D[DimState, np.dtype[npDType]]  # x_{i, t} \in R^{d_x}
States: TypeAlias = TypedList[NumPoints, State[DimState]]
StatesCollection: TypeAlias = TypedList[NumDemos, States[NumPoints, DimState]]

Action: TypeAlias = Array1D[DimAction, np.dtype[npDType]]  # a_{i, t} \in R^{d_a}
Actions: TypeAlias = TypedList[NumPoints, Action[DimAction]]
ActionsCollection: TypeAlias = TypedList[NumDemos, Actions[NumPoints, DimAction]]

Phase: TypeAlias = npDType  # tau \in [0, 1]
Phases: TypeAlias = TypedList[NumPoints, Phase]
PhasesCollection: TypeAlias = TypedList[NumDemos, Phases[NumPoints]]

Residual: TypeAlias = npDType  # r_{i, t}
Residuals: TypeAlias = TypedList[NumPoints, Residual]
ResidualsCollection: TypeAlias = TypedList[NumDemos, Residuals[NumPoints]]

ZScore: TypeAlias = npDType  # z_{i, t}
ZScores: TypeAlias = TypedList[NumPoints, ZScore]
ZScoresCollection: TypeAlias = TypedList[NumDemos, ZScores[NumPoints]]

TrustValue: TypeAlias = npDType  # w_{i, t}
TrustValues: TypeAlias = TypedList[NumPoints, TrustValue]
TrustValuesCollection: TypeAlias = TypedList[NumDemos, TrustValues[NumPoints]]

DemoIndex: TypeAlias = int  # i \in {0, 1, ..., N-1}
DemoIndices: TypeAlias = TypedList[NumPoints, DemoIndex]
DemoIndicesCollection: TypeAlias = TypedList[NumDemos, DemoIndices[NumPoints]]

TimeIndex: TypeAlias = int  # t \in {0, 1, ..., T_i-1}
TimeIndices: TypeAlias = TypedList[NumPoints, TimeIndex]
TimeIndicesCollection: TypeAlias = TypedList[NumDemos, TimeIndices[NumPoints]]


class SampleIndex(NamedTuple):  # (i, t)
    demo: DemoIndex  # i
    time: TimeIndex  # t


SampleIndices: TypeAlias = TypedList[NumPoints, SampleIndex]
SampleIndicesCollection: TypeAlias = TypedList[NumDemos, SampleIndices[NumPoints]]

BinIndex: TypeAlias = int  # b \in {0, 1, ..., B-1}

## ── Base ─────────────────────────────────────────────────────────────────────


# (x, a)
@dataclass
class StateActionPair(Generic[DimState, DimAction]):
    """A container for a State-Action pair."""

    state: State[DimState]  # x
    action: Action[DimAction]  # a


# (x_{i, t}, a_{i, t})
@dataclass
class Sample(StateActionPair[DimState, DimAction]):
    """A State-Action pair in the context of Demonstrations."""

    index: SampleIndex  # (i, t)


@dataclass
class Samples(Generic[NumPoints, DimState, DimAction]):
    samples: TypedList[NumPoints, Sample[DimState, DimAction]] = field(
        default_factory=TypedList[NumPoints, Sample[DimState, DimAction]]
    )  # [(x_{t}, a_{t})]_{t = 1}^{T}

    def __len__(self) -> NumPoints:
        return self.samples.length  # T

    @enforce_shapes
    def __getitem__(
        self,
        index: TimeIndex,  # t
        /,
    ) -> Sample[DimState, DimAction]:
        return self.samples[index]  # (x_{t}, a_{t})

    @enforce_shapes
    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        yield from self.samples

    @property
    def time_indices(self) -> TimeIndices[NumPoints]:
        return TimeIndices[NumPoints](range(self.__len__()))

    @enforce_shapes
    def append(self, sample: Sample[DimState, DimAction]) -> None:
        self.samples.append(sample)

    @enforce_shapes
    def states(self) -> States[NumPoints, DimState]:
        return States[NumPoints, DimState](sample.state for sample in self.samples)

    @enforce_shapes
    def actions(self) -> Actions[NumPoints, DimAction]:
        return Actions[NumPoints, DimAction](sample.action for sample in self.samples)

    @property
    def time_indices_and_samples(
        self,
    ) -> Iterator[tuple[TimeIndex, Sample[DimState, DimAction]]]:
        # ) -> TypedList[NumPoints, tuple[TimeIndex, Sample[DimState, DimAction]]]:
        for t in self.time_indices:
            sample = self.samples[t]
            yield (t, sample)
        # return TypedList[NumPoints, tuple[TimeIndex, Sample[DimState, DimAction]]](
        #     (sample.index.time, sample) for sample in self.samples
        # )

    @property
    def time_indices_and_states(
        self,
    ) -> Iterator[tuple[TimeIndex, State[DimState]]]:
        for t, sample in self.time_indices_and_samples:
            yield (t, sample.state)

    @property
    def time_indices_and_actions(
        self,
    ) -> Iterator[tuple[TimeIndex, Action[DimAction]]]:
        for t, sample in self.time_indices_and_samples:
            yield (t, sample.action)


@dataclass
class SamplesCollection(Generic[NumDemos, NumPoints, DimState, DimAction]):
    """A collection of state-action pairs (a sequence of state-action pair)."""

    collection: TypedList[NumDemos, Samples[NumPoints, DimState, DimAction]] = field(
        default_factory=TypedList[NumDemos, Samples[NumPoints, DimState, DimAction]]
    )  # [[(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}]_{i = 1}^{N}

    def __len__(self) -> NumDemos:
        return self.collection.length  # N

    @enforce_shapes
    @overload
    def __getitem__(
        self,
        index: DemoIndex,  # i
        /,
    ) -> Samples[
        NumPoints, DimState, DimAction
    ]: ...  # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
    @overload
    def __getitem__(  # ty: ignore[invalid-overload]
        self,
        index: SampleIndex,  # (i, t)
        /,
    ) -> Sample[DimState, DimAction]: ...  # (x_{i, t}, a_{i, t})
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex, /
    ) -> Samples[NumPoints, DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case SampleIndex(i, t):
                return self.collection[i][t]
            case DemoIndex() as i:
                return self.collection[i]
        raise IndexError

    @enforce_shapes
    def __iter__(self) -> Iterator[Samples[NumPoints, DimState, DimAction]]:
        yield from self.collection

    @property
    def demo_indices(self) -> DemoIndices[NumDemos]:
        return DemoIndices[NumDemos](range(self.__len__()))

    @enforce_shapes
    def samples(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Iterator[Sample[DimState, DimAction]]:  # (N x T_) or (N-1 x T_)
        for i, samples in enumerate(self.collection):
            if i == LOO_demo_index:
                continue
            yield from samples

    @enforce_shapes
    def states(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> States[int, DimState]:
        # (N x T_) or (N-1 x T_)
        return States[int, DimState](
            sample.state for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )

    @enforce_shapes
    def actions(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Actions[int, DimAction]:
        # (N x T_) or (N-1 x T_)
        return Actions[int, DimAction](
            sample.action for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )

    @property
    def iter_sample_indices_and_samples(
        self,
    ) -> Iterator[tuple[SampleIndex, Sample[DimState, DimAction]]]:
        for i in self.demo_indices:
            samples = self[i]
            for t in samples.time_indices:
                sample_idx = SampleIndex(i, t)
                sample = self[sample_idx]
                yield (sample_idx, sample)

    @property
    def iter_sample_indices_and_states(
        self,
    ) -> Iterator[tuple[SampleIndex, State[DimState]]]:
        for sample_idx, sample in self.iter_sample_indices_and_samples:
            yield (sample_idx, sample.state)

    @property
    def iter_sample_indices_and_actions(
        self,
    ) -> Iterator[tuple[SampleIndex, Action[DimAction]]]:
        for sample_idx, sample in self.iter_sample_indices_and_samples:
            yield (sample_idx, sample.action)


# Behaves like Samples
@dataclass(kw_only=True)
class Demonstration(Generic[NumPoints, DimState, DimAction]):  # D_i
    index: DemoIndex  # i
    states: States[NumPoints, DimState]  # [x_{i, t}]_{t = 1}^{T_i}
    actions: Actions[NumPoints, DimAction]  # [a_{i, t}]_{t = 1}^{T_i}

    def __post_init__(self) -> None:
        assert self.states.length == self.actions.length

    def __len__(self) -> NumPoints:
        assert self.states.length == self.actions.length
        return self.states.length  # T_i

    @enforce_shapes
    def __getitem__(
        self, index: TimeIndex, /
    ) -> Sample[DimState, DimAction]:  # (x_{i, t}, a_{i, t})
        return Sample(
            index=SampleIndex(demo=self.index, time=index),
            state=self.states[index],
            action=self.actions[index],
        )

    @enforce_shapes
    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
        for t in self.time_indices:
            yield self[t]

    @property
    def time_indices(self) -> TimeIndices[NumPoints]:
        return TimeIndices[NumPoints](range(self.__len__()))

    @property
    def state_dim(self) -> DimState:  # d_x
        return self.states[0].shape[0]

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.actions[0].shape[0]

    def sample(self, t: TimeIndex, /) -> Sample[DimState, DimAction]:
        return self[t]  # (x_{i, t}, a_{i, t})

    def samples(self) -> Samples[NumPoints, DimState, DimAction]:
        return Samples(
            TypedList[NumPoints, Sample[DimState, DimAction]](sample for sample in self)
        )


# Behaves like SamplesCollection
@dataclass(slots=True)
class Demonstrations(
    Generic[NumDemos, NumPoints, DimState, DimAction]
):  # [D_i]_{i = 1}^{N}
    demos: TypedList[NumDemos, Demonstration[NumPoints, DimState, DimAction]]

    def __len__(self) -> NumDemos:
        return self.demos.length  # N

    @enforce_shapes
    @overload
    def __getitem__(
        self, index: DemoIndex, /
    ) -> Demonstration[NumPoints, DimState, DimAction]: ...
    @overload
    def __getitem__(self, index: SampleIndex, /) -> Sample[DimState, DimAction]: ...  # ty: ignore[invalid-overload]
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex, /
    ) -> Demonstration[NumPoints, DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case SampleIndex(i, t):
                return self.demos[i][t]
            case DemoIndex() as i:
                return self.demos[i]
        raise IndexError

    @enforce_shapes
    def __iter__(self) -> Iterator[Demonstration[NumPoints, DimState, DimAction]]:
        yield from self.demos  # D_i

    @property
    def demo_indices(self) -> DemoIndices[NumDemos]:
        return DemoIndices[NumDemos](range(self.__len__()))

    @property
    def state_dim(self) -> DimState:  # d_x
        return self.demos[0].state_dim

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.demos[0].action_dim


## ── Phase Alignment ──────────────────────────────────────────────────────────


class PhaseScorer(nn.Module, Generic[DimState]):
    """A small neural network (MLP) to estimate state-dependent phase score `g_psi`."""

    def __init__(self, state_dim: DimState, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        states: Tensor,  # (batch, state_dim)
    ) -> Tensor:  # (batch,) unnormalized phase scores
        forward: Tensor = self.network(states)
        return forward.squeeze(-1)


@dataclass
class PhaseEstimator(Generic[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    device: torch.device = field(kw_only=True, default_factory=get_torch_device_auto)
    ##
    scorer: PhaseScorer[DimState] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def compute_ranking_loss(self, margin: float = 1.0) -> Tensor:  # L_rank
        loss = torch.tensor(0.0, device=self.device)
        for demo in self.demonstrations:
            states = Tensor(np.array(demo.states)).float().to(self.device)
            scores: Tensor = self.scorer(states)  # (T_i,)
            diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (T_i, T_i)
            mask = torch.ones_like(diff).triu(diagonal=1)  # Enforces `t > t'`
            loss_matrix = F.softplus(margin - diff) * mask
            loss += loss_matrix.sum() / (mask.sum() + EPS)  # Normalise over valid pairs
        if (n_demos := self.demonstrations.__len__()) > 0:
            loss /= n_demos  # Normalise over demonstrations
        return loss

    def train(
        self,
        *,
        hidden_dim: int = 128,
        margin: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 240,
    ) -> Tensor:
        state_dim = self.demonstrations.state_dim
        scorer = PhaseScorer(state_dim=state_dim, hidden_dim=hidden_dim)
        self.scorer = scorer.to(self.device)
        self.optimiser = torch.optim.Adam(self.scorer.parameters(), lr=lr)

        self.scorer.train()
        loss = self.compute_ranking_loss(margin=margin)
        for _epoch in track(range(epochs), description="[bold]Phase training[/]"):
            self.optimiser.zero_grad()
            loss = self.compute_ranking_loss(margin=margin)
            loss.backward()  # type: ignore[no-untyped-call]  # ty: ignore[unused-ignore-comment]
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), 1.0)
            self.optimiser.step()  # pyright: ignore[reportUnknownMemberType]
        return loss

    @enforce_shapes
    def estimate_phases(self) -> PhasesCollection[NumDemos, NumPoints]:
        # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        self.scorer.eval()
        phases = PhasesCollection[NumDemos, NumPoints]()
        with torch.no_grad():
            for demo in self.demonstrations:
                states = Tensor(np.array(demo.states)).float().to(self.device)
                scores: Tensor = self.scorer(states)
                _scores = scores.cpu().numpy()
                normalised = Phases[NumPoints](normalise(_scores, method="MINMAX"))
                phases.append(normalised)
        return phases


## ── PACER ────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class RobustStatistics(Generic[DimState, DimAction]):
    """Robust consensus statistics for a set of samples."""

    ## Stable anchors
    median_action: Action[DimAction]  # alpha_a[b] = median{ a_{i, t} : (i, t) \in I_b }
    median_state: State[DimState]  # alpha_s[b] = median{ x_{i, t} : (i, t) \in I_b }

    ## Pace
    median_action_strength: npDType
    # beta_a[b] = median{ ||a_{i, t}|| : (i, t) \in I_b }
    # Captures strength of actions
    median_state_change: npDType
    # beta_s[b] = median{ ||xdot_{i, t}|| : (i, t) \in I_b }
    # Captures typical rate of state change

    ## Local task dynamics
    # NOTE: `action_tangent` and `state_tangent` are not stored here,
    # but instead, computed and stored in RibbonToken.


@dataclass(kw_only=True)
class RibbonToken(Generic[DimState, DimAction]):  # z_b
    """
    Robust structured descriptor for a single phase bin `b`.\\
    Encodes both consensus behaviour and degree of variability present at phase `b`.
    """

    median_action: Action[DimAction]  # alpha_a[b]
    median_action_strength: npDType  # beta_a[b]
    median_state: State[DimState]  # alpha_s[b]
    median_state_change: npDType  # beta_s[b]

    ## Local task dynamics
    action_tangent: Action[DimAction] = field(init=False)
    # t_a[b] <- diff{ alpha_a[b] }
    state_tangent: State[DimState] | None = field(init=False)
    # t_s[b] <- diff{ alpha_s[b] }

    MAD_action_residual: Residual  # Median Absolute Deviation of action residuals


@dataclass(kw_only=True)
class Bin(Generic[NumDemos, NumPoints, DimState, DimAction]):
    index: BinIndex  # b
    samples_collection: SamplesCollection[NumDemos, NumPoints, DimState, DimAction] = (
        field(
            default_factory=SamplesCollection[NumDemos, NumPoints, DimState, DimAction]
        )
    )
    ##
    ribbon_token: RibbonToken[DimState, DimAction] = field(init=False)

    def samples(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Samples[int, DimState, DimAction]:
        # (N x T_) or (N-1 x T_)
        return Samples(
            TypedList[int, Sample[DimState, DimAction]](
                self.samples_collection.samples(LOO_demo_index=LOO_demo_index)
            )
        )

    def states(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> States[int, DimState]:
        # (N x T_) or (N-1 x T_)
        return self.samples_collection.states(LOO_demo_index=LOO_demo_index)

    def actions(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Actions[int, DimAction]:
        # (N x T_) or (N-1 x T_)
        return self.samples_collection.actions(LOO_demo_index=LOO_demo_index)


@dataclass
class PACER(Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    phase_estimator: PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
    n_bins: NumBins = field(default=cast(NumBins, 96), kw_only=True)  # B
    ##
    bins: TypedList[NumBins, Bin[NumDemos, NumPoints, DimState, DimAction]] = field(
        init=False
    )
    phases: PhasesCollection[NumDemos, NumPoints] = field(init=False)

    @property
    def demonstrations(
        self,
    ) -> Demonstrations[NumDemos, NumPoints, DimState, DimAction]:
        return self.phase_estimator.demonstrations

    def phase_range(self, bin_idx: BinIndex) -> tuple[Phase, Phase]:
        return (Phase(bin_idx / self.n_bins), Phase((bin_idx + 1) / self.n_bins))

    def sample_index_to_bin_index(self, sample_idx: SampleIndex) -> BinIndex:
        i, t = sample_idx
        tau: Phase = Phase(self.phases[i][t])
        bin_idx: BinIndex = min(int(tau * self.n_bins), self.n_bins - 1)
        assert 0 <= bin_idx < self.n_bins
        return bin_idx

    @property
    def sample_indices(self) -> Iterator[SampleIndex]:
        i: DemoIndex
        t: TimeIndex
        for i in range(self.phases.__len__()):
            for t in range(self.phases[i].__len__()):
                sample_idx = SampleIndex(i, t)
                yield sample_idx

    def make_bins(self) -> None:
        self.phases = self.phase_estimator.estimate_phases()
        N = self.demonstrations.__len__()
        self.bins = TypedList[
            NumBins, Bin[NumDemos, NumPoints, DimState, DimAction]
        ].full(
            self.n_bins,
            lambda bin_idx: Bin[NumDemos, NumPoints, DimState, DimAction](
                index=bin_idx,
                samples_collection=SamplesCollection(
                    collection=TypedList[
                        NumDemos, Samples[NumPoints, DimState, DimAction]
                    ].full(N, Samples[NumPoints, DimState, DimAction]()),
                ),
            ),
        )
        for sample_idx in self.sample_indices:
            demo_idx, _ = sample_idx
            bin_idx = self.sample_index_to_bin_index(sample_idx)
            bin = self.bins[bin_idx]
            sample = self.demonstrations[sample_idx]
            collection = bin.samples_collection[demo_idx]
            collection.append(sample)

    @enforce_shapes
    def compute_robust_consensus_statistics(
        self, samples: Samples[Any, DimState, DimAction]
    ) -> RobustStatistics[DimState, DimAction]:
        states = samples.states()
        actions = samples.actions()
        action_norms = list(la.norm(action) for action in actions)
        state_change_norms = [
            la.norm(states[t + 1] - states[t]) for t in range(len(states) - 1)
        ]

        median_action = Action[DimAction](median(actions, axis=0))
        median_state = State[DimState](median(states, axis=0))
        median_action_strength = npDType(median(action_norms, axis=0))
        median_state_change = npDType(median(state_change_norms, axis=0))

        return RobustStatistics(
            median_action=median_action,
            median_state=median_state,
            median_action_strength=median_action_strength,
            median_state_change=median_state_change,
        )

    @enforce_shapes
    def compute_z_scores(self) -> ZScoresCollection[NumDemos, NumPoints]:  # (N x T_)
        N = self.demonstrations.__len__()

        # (N x T_)
        z_scores = ZScoresCollection[NumDemos, NumPoints].full(
            N,
            lambda i: ZScores[NumPoints](
                [ZScore(0) for _ in self.demonstrations[i].time_indices]
            ),
        )

        for bin in self.bins:
            for i in self.demonstrations.demo_indices:
                loo_samples = bin.samples(LOO_demo_index=i)
                loo_stats = self.compute_robust_consensus_statistics(loo_samples)
                loo_median_action = loo_stats.median_action  # alpha_a^{(-i)}[b]

                loo_residuals = [
                    Residual(la.norm(action - loo_median_action))
                    for action in bin.actions(LOO_demo_index=i)
                ]
                median_residual = Residual(median(loo_residuals))
                abs_deviations = [
                    Residual(abs(residual - median_residual))
                    for residual in loo_residuals
                ]
                MAD_residual = Residual(
                    MAD_SCALE * median(abs_deviations)
                )  # MAD_{a}^{(-i)}[b]

                demo_samples = bin.samples_collection[i]
                for t, action in demo_samples.time_indices_and_actions:
                    self_residual = Residual(
                        la.norm(action - loo_median_action)
                    )  # r^{-i}_{i, t}
                    z_score = ZScore(self_residual / (MAD_residual + EPS))  # z_{i, t}
                    z_scores[i][t] = z_score

        return z_scores

    @enforce_shapes
    def compute_trust_values(
        self,
        *,
        cutoff: npDType | float,  # c
        min_trust: npDType | float,  # w_min
    ) -> TrustValuesCollection[NumDemos, NumPoints]:  # (N x T_)
        assert 3 <= cutoff <= 5
        N = self.demonstrations.__len__()
        trust_values = TrustValuesCollection[NumDemos, NumPoints].full(
            N, TrustValues[NumPoints]()
        )  # (N x T_)
        z_scores = self.compute_z_scores()
        for i, scores in enumerate(z_scores):
            for _t, z_score in enumerate(scores):
                if z_score <= cutoff:
                    trust_value = (1 - (z_score / cutoff) ** 2) ** 2
                else:
                    trust_value = TrustValue(0)
                if trust_value < min_trust:
                    trust_value = TrustValue(min_trust)
                trust_values[i].append(trust_value)
        return trust_values  # [[w_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}

    @enforce_shapes
    def consolidate_ribbon_tokens(self) -> None:
        bin_median_actions = Actions[NumPoints, DimAction]()
        bin_median_states = States[NumPoints, DimState]()

        for bin in self.bins:
            stats = self.compute_robust_consensus_statistics(bin.samples())

            bin_median_action = stats.median_action  # alpha_a[b]
            bin_action_residuals = list[Residual]()
            for action in bin.actions():
                residual = Residual(la.norm(action - bin_median_action))  # r_{i, t}
                bin_action_residuals.append(residual)
            bin_median_action_residual = Residual(median(bin_action_residuals))
            abs_deviations = list[Residual]()
            for residual in bin_action_residuals:
                abs_deviation = Residual(abs(residual - bin_median_action_residual))
                abs_deviations.append(abs_deviation)
            MAD_action_residual = Residual(MAD_SCALE * median(abs_deviations))

            bin.ribbon_token = RibbonToken(
                median_action=stats.median_action,
                median_action_strength=stats.median_action_strength,
                median_state=stats.median_state,
                median_state_change=stats.median_state_change,
                MAD_action_residual=MAD_action_residual,
            )
            bin_median_actions.append(stats.median_action)
            bin_median_states.append(stats.median_state)

        for b, bin in enumerate(self.bins):
            if b == 0:
                p, q, f = b + 1, b, 1.0
            elif b == self.n_bins - 1:
                p, q, f = b, b - 1, 1.0
            else:
                p, q, f = b, b - 1, 0.5
            action_tangent = Action[DimAction](
                f * (bin_median_actions[p] - bin_median_actions[q])
            )
            state_tangent = State[DimState](
                f * (bin_median_states[p] - bin_median_states[q])
            )
            bin.ribbon_token.action_tangent = action_tangent
            bin.ribbon_token.state_tangent = state_tangent

    @enforce_shapes
    def compute_pseudo_labels(
        self,
        trust_values: TrustValuesCollection[NumDemos, NumPoints],
        *,
        debias_weight: npDType | float,  # lambda_{debias}
        sideways_attenuation_shrinkage: npDType | float = 0.5,  # rho_0
        speed_regularisation_influence: npDType | float = 0.5,  # eta_0
        temporal_smoothing_weight: npDType | float = 0.0,  # kappa
    ) -> ActionsCollection[NumDemos, NumPoints, DimAction]:  # (N x T_)
        N = self.demonstrations.__len__()
        pseudo_labels = ActionsCollection[NumDemos, NumPoints, DimAction].full(
            N, Actions[NumPoints, DimAction]()
        )
        _labels = ActionsCollection[NumDemos, NumPoints, DimAction].full(
            N, Actions[NumPoints, DimAction]()
        )  # [[y^{(3)}_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        self.consolidate_ribbon_tokens()
        rho_0 = sideways_attenuation_shrinkage
        assert 0 <= rho_0 <= 1
        eta_0 = speed_regularisation_influence
        assert 0 <= eta_0 <= 1
        kappa = temporal_smoothing_weight

        for bin in self.bins:
            # Alignment with ribbon tangent
            token = bin.ribbon_token  # z_b
            tangent = (
                token.state_tangent
                if token.state_tangent is not None
                else token.action_tangent
            )
            unit_tangent = Array1D[int](normalise(tangent, method="NORM"))  # t_{dir}[b]

            for j in self.demonstrations.demo_indices:
                loo_samples = bin.samples(LOO_demo_index=j)
                loo_stats = self.compute_robust_consensus_statistics(loo_samples)
                bin_median_action = loo_stats.median_action  # alpha_a^{(-j)}[b]

                demo_samples = bin.samples_collection[j]
                for t, action in demo_samples.time_indices_and_actions:
                    w = trust_values[j][t]  # w_{i, t}

                    # Debiasing towards the anchor
                    gamma = 1 - debias_weight * (1 - w)  # gamma_{i, t}
                    assert 0 <= gamma <= 1
                    y1 = Action[DimAction](
                        gamma * action + (1 - gamma) * bin_median_action
                    )  # y^{(1)}_{i, t}

                    # Sideways attentuation
                    y1_pll = Action[DimAction](np.dot(y1, unit_tangent) * unit_tangent)
                    y1_perp = Action[DimAction](y1 - y1_pll)
                    has_state_tangent = token.state_tangent is not None
                    rho = rho_0 * (1 - w) if has_state_tangent else npDType(0)
                    y2 = Action[DimAction](
                        y1_pll + (1 - rho) * y1_perp
                    )  # y^{(2)}_{i, t}

                    # Speed regularisation
                    eta = eta_0 * (1 - w)  # eta_{i, t}
                    beta_a = bin.ribbon_token.median_action_strength
                    s = (1 - eta) * la.norm(y2) + eta * beta_a  # s_{i, t}
                    y3 = Action[DimAction](
                        s * (y2 / (la.norm(y2) + EPS)), dtype=npDType
                    )  # y^{(3)}_{i, t}

                    _labels[j].append(y3)

        # Temporal smoothing
        for i in self.demonstrations.demo_indices:
            ystar_prev: Action[DimAction] | None = None
            for t in self.demonstrations.demos[i].time_indices:
                y3 = _labels[i][t]
                if ystar_prev is None:  # t = 0
                    ystar_prev = y3
                ystar = Action[DimAction](
                    (1 - kappa) * y3 + kappa * ystar_prev, dtype=npDType
                )
                pseudo_labels[i].append(ystar)
                ystar_prev = ystar

        return pseudo_labels


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
