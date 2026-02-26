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

import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeAlias, TypeVar, overload

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import optype.numpy as onp
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch import Tensor
from typed_numpy._typed.context import enforce_shapes
from typed_numpy._typed.helpers import Array1D

from pacer import console

## ── Typings ──────────────────────────────────────────────────────────────────

npDType: TypeAlias = np.float32
torchDType = torch.float32

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i

DemoIndex: TypeAlias = int  # i \in {0, 1, ..., N-1}
TimeIndex: TypeAlias = int  # t \in {0, 1, ..., T_i-1}
BinIndex: TypeAlias = int  # b \in {0, 1, ..., B-1}

State: TypeAlias = Array1D[DimState, np.dtype[npDType]]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = Array1D[DimAction, np.dtype[npDType]]  # a_{i, t} \in R^{d_a}
States: TypeAlias = list[State[DimState]]
Actions: TypeAlias = list[Action[DimAction]]
StatesCollection: TypeAlias = list[States[DimState]]
ActionsCollection: TypeAlias = list[Actions[DimAction]]

Phase: TypeAlias = npDType  # tau \in [0, 1]
Phases: TypeAlias = list[Phase]
PhasesCollection: TypeAlias = list[Phases]

SampleIndex: TypeAlias = tuple[DemoIndex, TimeIndex]  # (i, t)
SampleIndices: TypeAlias = list[SampleIndex]

## ── Utils ────────────────────────────────────────────────────────────────────

SEED = 42
EPS: float = 1e-8
MAD_SCALE: float = 1.4826  # Gaussian consistency factor for MAD


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def get_torch_device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        torch_device_auto = torch.device("mps")
    elif torch.cuda.is_available():
        torch_device_auto = torch.device("cuda")
    else:
        torch_device_auto = torch.device("cpu")
    console.print(f"Using device: [green]{torch_device_auto}[/]")
    return torch_device_auto


set_seed(SEED)


def median(
    arr: npt.ArrayLike, /, axis: int | Sequence[int] | None = None
) -> np.ndarray:
    arr = np.asarray(arr)
    return np.median(arr, axis=axis)  # type: ignore[no-any-return]  # ty: ignore[unused-ignore-comment]


def normalise(
    vec: onp.ToArray1D, /, method: Literal["NORM", "MINMAX", "ZSCORE"]
) -> np.ndarray:
    vec = np.asarray(vec, dtype=npDType)
    match method:
        case "NORM":
            norm = la.norm(vec)
            return vec / (norm + EPS)  # type: ignore[no-any-return]  # ty: ignore[unused-ignore-comment]
        case "MINMAX" | "ZSCORE":
            min_: float = vec.min()
            max_: float = vec.max()
            return (vec - min_) / (max_ - min_ + EPS)


## ── Base ─────────────────────────────────────────────────────────────────────


# (x_{i, t}, a_{i, t})
@dataclass
class Sample(Generic[DimState, DimAction]):
    """A container for a State-Action pair"""

    state: State[DimState]  # x_{i, t}
    action: Action[DimAction]  # a_{i, t}


@dataclass
class Samples(Generic[DimState, DimAction]):
    samples: list[Sample[DimState, DimAction]] = field(
        default_factory=list[Sample[DimState, DimAction]]
    )  # [(x_{t}, a_{t})]_{t = 1}^{T}

    def __len__(self) -> int:
        return len(self.samples)  # T

    @enforce_shapes
    def __getitem__(
        self,
        index: TimeIndex,  # t
    ) -> Sample[DimState, DimAction]:
        return self.samples[index]  # (x_{t}, a_{t})

    @enforce_shapes
    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        for sample in self.samples:
            yield sample

    @enforce_shapes
    def append(self, sample: Sample[DimState, DimAction]) -> None:
        self.samples.append(sample)

    @enforce_shapes
    def extend(self, samples: Iterable[Sample[DimState, DimAction]]) -> None:
        self.samples.extend(samples)

    @enforce_shapes
    def states(self) -> States[DimState]:
        return list(sample.state for sample in self.samples)

    @enforce_shapes
    def actions(self) -> Actions[DimAction]:
        return list(sample.action for sample in self.samples)


@dataclass
class SamplesCollection(Generic[DimState, DimAction]):
    """A collection of state-action pairs (a sequence of state-action pair)."""

    collection: list[Samples[DimState, DimAction]] = field(
        default_factory=list[Samples[DimState, DimAction]]
    )  # [[(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}]_{i = 1}^{N}

    def __len__(self) -> int:
        return len(self.collection)  # N

    @enforce_shapes
    @overload
    def __getitem__(
        self,
        index: DemoIndex,  # i
    ) -> Samples[DimState, DimAction]: ...  # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
    @overload
    def __getitem__(  # ty: ignore[invalid-overload]
        self,
        index: SampleIndex,  # (i, t)
    ) -> Sample[DimState, DimAction]: ...  # (x_{i, t}, a_{i, t})
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex
    ) -> Samples[DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case tuple():
                i, t = index
                return self.collection[i][t]
            case DemoIndex():
                return self.collection[index]
        raise IndexError

    @enforce_shapes
    def __iter__(self) -> Iterator[Samples[DimState, DimAction]]:
        for samples in self.collection:
            yield samples

    @enforce_shapes
    def append(self, samples: Samples[DimState, DimAction]) -> None:
        self.collection.append(samples)

    @enforce_shapes
    def extend(self, samples: Iterable[Samples[DimState, DimAction]]) -> None:
        self.collection.extend(samples)

    @enforce_shapes
    def samples(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Iterator[Sample[DimState, DimAction]]:  # (N x T_) or (N-1 x T_)
        for i, samples in enumerate(self.collection):
            if i == LOO_demo_index:
                continue
            for sample in samples:
                yield sample

    @enforce_shapes
    def states(self, *, LOO_demo_index: DemoIndex | None = None) -> States[DimState]:
        # (N x T_) or (N-1 x T_)
        return list(
            sample.state for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )

    @enforce_shapes
    def actions(self, *, LOO_demo_index: DemoIndex | None = None) -> Actions[DimAction]:
        # (N x T_) or (N-1 x T_)
        return list(
            sample.action for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )


# Behaves like Samples
@dataclass(kw_only=True)
class Demonstration(Generic[DimState, DimAction]):  # D_i
    index: DemoIndex  # i
    states: States[DimState]  # [x_{i, t}]_{t = 1}^{T_i}
    actions: Actions[DimAction]  # [a_{i, t}]_{t = 1}^{T_i}

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions)

    def __len__(self) -> int:
        return len(self.states)  # T_i

    @enforce_shapes
    def __getitem__(
        self, t: TimeIndex, /
    ) -> Sample[DimState, DimAction]:  # (x_{i, t}, a_{i, t})
        return Sample(self.states[t], self.actions[t])

    @enforce_shapes
    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
        for t in range(len(self)):
            yield self[t]

    @property
    def state_dim(self) -> DimState:  # d_x
        return self.states[0].shape[0]

    @property
    def action_dim(self) -> DimAction:  # d_a
        return self.actions[0].shape[0]

    def sample(self, t: TimeIndex, /) -> Sample[DimState, DimAction]:
        return self[t]  # (x_{i, t}, a_{i, t})

    def samples(self) -> Samples[DimState, DimAction]:
        return Samples(list(sample for sample in self))


# Behaves like SamplesCollection
@dataclass(slots=True)
class Demonstrations(Generic[DimState, DimAction]):  # [D_i]_{i = 1}^{N}
    demos: list[Demonstration[DimState, DimAction]]

    def __len__(self) -> int:
        return len(self.demos)

    @enforce_shapes
    @overload
    def __getitem__(self, index: DemoIndex) -> Demonstration[DimState, DimAction]: ...
    @overload
    def __getitem__(self, index: SampleIndex) -> Sample[DimState, DimAction]: ...  # ty: ignore[invalid-overload]
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex
    ) -> Demonstration[DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case tuple():
                i, t = index
                return self.demos[i][t]
            case DemoIndex():
                return self.demos[index]
        raise IndexError

    @enforce_shapes
    def __iter__(self) -> Iterator[Demonstration[DimState, DimAction]]:
        for demo in self.demos:
            yield demo  # D_i

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
class PhaseEstimator(Generic[DimState, DimAction]):
    demonstrations: Demonstrations[DimState, DimAction]
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
            loss += loss_matrix.mean()
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
    def estimate_phases(
        self,
    ) -> list[Array1D[int]]:  # [[tau_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
        self.scorer.eval()
        phases = list[Array1D[int]]()
        with torch.no_grad():
            for demo in self.demonstrations:
                states = Tensor(np.array(demo.states)).float().to(self.device)
                scores: Tensor = self.scorer(states)
                _scores = scores.cpu().numpy()
                normalised = Array1D(normalise(_scores, method="MINMAX"))
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

    MAD_action_residual: npDType  # Median Absolute Deviation of action residuals


@dataclass(kw_only=True)
class Bin(Generic[DimState, DimAction]):
    index: BinIndex  # b
    samples_collection: SamplesCollection[DimState, DimAction] = field(
        default_factory=SamplesCollection[DimState, DimAction]
    )
    ##
    ribbon_token: RibbonToken[DimState, DimAction] = field(init=False)

    def samples(
        self, *, LOO_demo_index: DemoIndex | None = None
    ) -> Samples[DimState, DimAction]:
        # (N x T_) or (N-1 x T_)
        return Samples(
            list(self.samples_collection.samples(LOO_demo_index=LOO_demo_index))
        )

    def states(self, *, LOO_demo_index: DemoIndex | None = None) -> States[DimState]:
        # (N x T_) or (N-1 x T_)
        return self.samples_collection.states(LOO_demo_index=LOO_demo_index)

    def actions(self, *, LOO_demo_index: DemoIndex | None = None) -> Actions[DimAction]:
        # (N x T_) or (N-1 x T_)
        return self.samples_collection.actions(LOO_demo_index=LOO_demo_index)


@dataclass
class PACER(Generic[DimState, DimAction]):
    phase_estimator: PhaseEstimator[DimState, DimAction]
    n_bins: int = field(default=96, kw_only=True)  # B
    ##
    bins: list[Bin[DimState, DimAction]] = field(init=False)

    @property
    def demonstrations(self) -> Demonstrations[DimState, DimAction]:
        return self.phase_estimator.demonstrations

    def phase_range(self, bin_idx: BinIndex) -> tuple[Phase, Phase]:
        return (Phase(bin_idx / self.n_bins), Phase((bin_idx + 1) / self.n_bins))

    def make_bins(self) -> None:
        phases = self.phase_estimator.estimate_phases()
        self.bins = [
            Bin[DimState, DimAction](
                index=bin_idx,
                samples_collection=SamplesCollection(
                    collection=[Samples() for _ in range(len(self.demonstrations))]
                ),
            )
            for bin_idx in range(self.n_bins)
        ]
        for i in range(len(phases)):
            for t in range(len(phases[i])):
                tau: Phase = Phase(phases[i][t])
                bin_idx: BinIndex = min(int(tau * self.n_bins), self.n_bins - 1)
                assert bin_idx < self.n_bins
                bin = self.bins[bin_idx]
                sample_idx: SampleIndex = (i, t)
                sample = self.demonstrations[sample_idx]
                collection = bin.samples_collection.collection[i]
                collection.append(sample)

    @enforce_shapes
    def compute_robust_consensus_statistics(
        self, samples: Samples[DimState, DimAction]
    ) -> RobustStatistics[DimState, DimAction]:
        states = samples.states()
        actions = samples.actions()
        action_norms = list(la.norm(action) for action in actions)
        state_change_norms = list(la.norm(np.diff(state)) for state in states)

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
    def compute_z_scores(self) -> PhasesCollection:  # (N x T_)
        N = len(self.demonstrations)

        # (N x T_)
        # [[r^{(-i)}_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}]
        self_action_residuals = [list[npDType]() for _ in range(N)]

        # (N,)
        MAD_residuals = list[npDType]()

        # (N x T_)
        z_scores = [list[npDType]() for _ in range(N)]

        for bin in self.bins:
            for j in range(N):
                loo_samples = bin.samples(LOO_demo_index=j)
                loo_stats = self.compute_robust_consensus_statistics(loo_samples)
                bin_median_action = loo_stats.median_action  # alpha_a^{(-j)}[b]

                demo_samples = bin.samples_collection.collection[j]
                for action in demo_samples.actions():
                    residual = npDType(
                        la.norm(action - bin_median_action)
                    )  # r^{-i}_{i, t}
                    self_action_residuals[j].append(residual)

                bin_action_residuals = list[npDType]()  # LOO
                for action in bin.actions(LOO_demo_index=j):
                    residual = npDType(
                        la.norm(action - bin_median_action)
                    )  # r^{-j}_{i, t}
                    bin_action_residuals.append(residual)

                bin_median_action_residual = npDType(median(bin_action_residuals))
                abs_deviations = list[npDType]()
                for residual in bin_action_residuals:
                    abs_deviation = npDType(abs(residual - bin_median_action_residual))
                    abs_deviations.append(abs_deviation)
                MAD_residual = npDType(MAD_SCALE * median(abs_deviations))
                MAD_residuals.append(MAD_residual)

        for i in range(N):
            T_i = len(self.demonstrations.demos[i])
            denom = MAD_residuals[i] + EPS
            for t in range(T_i):
                z_score = (self_action_residuals[i][t]) / denom  # z_{i, t}
                z_scores[i].append(z_score)

        return z_scores

    @enforce_shapes
    def compute_trust_values(
        self,
        *,
        cutoff: npDType | float,  # c
        min_trust: npDType | float,  # w_min
    ) -> PhasesCollection:  # (N x T_)
        assert 3 <= cutoff <= 5
        N = len(self.demonstrations)
        trust_values = [list[npDType]() for _ in range(N)]  # (N x T_)
        z_scores = self.compute_z_scores()
        for i, scores in enumerate(z_scores):
            for _t, z_score in enumerate(scores):
                if z_score <= cutoff:
                    trust_value = (1 - (z_score / cutoff) ** 2) ** 2
                else:
                    trust_value = npDType(0)
                if trust_value < min_trust:
                    trust_value = npDType(min_trust)
                trust_values[i].append(trust_value)
        return trust_values  # [[w_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}

    @enforce_shapes
    def consolidate_ribbon_tokens(self) -> None:
        bin_median_actions = Actions[DimAction]()
        bin_median_states = States[DimState]()

        for bin in self.bins:
            stats = self.compute_robust_consensus_statistics(bin.samples())

            bin_median_action = stats.median_action  # alpha_a[b]
            bin_action_residuals = list[npDType]()
            for action in bin.actions():
                residual = npDType(la.norm(action - bin_median_action))  # r_{i, t}
                bin_action_residuals.append(residual)
            bin_median_action_residual = npDType(median(bin_action_residuals))
            abs_deviations = list[npDType]()
            for residual in bin_action_residuals:
                abs_deviation = npDType(abs(residual - bin_median_action_residual))
                abs_deviations.append(abs_deviation)
            MAD_action_residual = npDType(MAD_SCALE * median(abs_deviations))

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
        trust_values: PhasesCollection,
        *,
        debias_weight: npDType | float,  # lambda_{debias}
        sideways_attenuation_shrinkage: npDType | float = 0.5,  # rho_0
        speed_regularisation_influence: npDType | float = 0.5,  # eta_0
        temporal_smoothing_weight: npDType | float = 0.0,  # kappa
    ) -> ActionsCollection[DimAction]:  # (N x T_)
        N = len(self.demonstrations)
        pseudo_labels = [list[Action[DimAction]]() for _ in range(N)]
        _labels = [
            list[Action[DimAction]]() for _ in range(N)
        ]  # [[y^{(3)}_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}
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

            for j in range(N):
                loo_samples = bin.samples(LOO_demo_index=j)
                loo_stats = self.compute_robust_consensus_statistics(loo_samples)
                bin_median_action = loo_stats.median_action  # alpha_a^{(-j)}[b]

                demo_samples = bin.samples_collection.collection[j]
                for t, action in enumerate(demo_samples.actions()):
                    w = trust_values[j][t]  # w_{i, t}

                    # Debiasing towards the anchor
                    gamma = 1 - debias_weight * (1 - w)  # gamma_{i, t}
                    assert 0 <= gamma <= 1
                    y1 = Action[DimAction](
                        gamma * action + (1 - gamma) * bin_median_action
                    )  # y^{(1)}_{i, t}

                    # Sideways attentuation
                    y1_pll = Action[DimAction](np.dot(y1, unit_tangent) * tangent)
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
                        s * (y2 / la.norm(y2) + EPS), dtype=npDType
                    )  # y^{(3)}_{i, t}

                    _labels[j].append(y3)

        # Temporal smoothing
        for i in range(N):
            T_i = len(self.demonstrations.demos[i])
            ystar_prev: Action[DimAction] | None = None
            for t in range(T_i):
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
class PACERBCTrainer(Generic[DimState, DimAction]):
    """PACER + Behavioral cloning policy trainer"""

    demonstrations: Demonstrations[DimState, DimAction]
    device: torch.device = field(kw_only=True, default_factory=get_torch_device_auto)
    ##
    phase_estimator: PhaseEstimator[DimState, DimAction] = field(init=False)
    pacer: PACER[DimState, DimAction] = field(init=False)
    trust_values: PhasesCollection = field(init=False)
    pseudo_labels: ActionsCollection[DimAction] = field(init=False)
    policy: BCPolicy[DimState, DimAction] = field(init=False)
    optimiser: torch.optim.Optimizer = field(init=False)

    def prepare(
        self,
        *,
        phase_hidden_dim: int = 128,
        phase_margin: float = 1.0,
        phase_lr: float = 1e-3,
        phase_epochs: int = 240,
        n_bins: int = 96,
        tukey_cutoff: npDType | float = 4.685,  # c
        min_trust: npDType | float = 0.02,  # w_min
        debias_weight: npDType | float = 0.5,  # lambda_{debias}
        sideways_attenuation_shrinkage: npDType | float = 0.5,  # rho_0
        speed_regularisation_influence: npDType | float = 0.5,  # eta_0
        temporal_smoothing_weight: npDType | float = 0.0,  # kappa
    ) -> Tensor:
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

        if total_weight > 0:
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
    def predict(self, states: States[DimState]) -> Actions[DimAction]:
        self.policy.eval()
        with torch.no_grad():
            states_tensor = Tensor(np.array(states)).float().to(self.device)
            actions_tensor: Tensor = self.policy(states_tensor)
            actions_np = actions_tensor.cpu().numpy()
        actions = list(Action[DimAction](action_np) for action_np in actions_np)
        return actions


## ─────────────────────────────────────────────────────────────────────────────
