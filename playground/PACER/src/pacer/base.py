"""
PACER Base
=======
Implementation follows the following paper from
Shreyas Kumar & Ravi Prakash, CoRL 2025 Workshop on Robot Data:
"PACER: Progress-Aligned Curation for Error-Resilient Imitation Learning"
https://openreview.net/forum?id=gaYyBvP2Rz
"""
# src/pacer/base.py

import random
from dataclasses import dataclass, field
from typing import (
    Generic,
    Iterable,
    Iterator,
    Literal,
    Self,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import optype.numpy as onp
import torch
import torch.nn as nn
import torch.nn.functional as F
from deprecated import deprecated  # type: ignore
from torch import Tensor
from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.context import enforce_shapes  # type: ignore

from pacer import console

npDType: TypeAlias = np.float32
torchDType = torch.float32

Dim1 = TypeVar("Dim1", bound=int, default=int)
Array1D: TypeAlias = TypedNDArray[tuple[Dim1], np.dtype[npDType]]

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
State: TypeAlias = Array1D[DimState]  # x_{i, t} \in R^{d_x}
Action: TypeAlias = Array1D[DimAction]  # a_{i, t} \in R^{d_a}
type Phase = float  # tau \in [0, 1]
type DemoIndex = int  # i \i {0, 1, ..., N-1}
type TimeIndex = int  # t \i {0, 1, ..., T_i-1}
type BinIndex = int  # b \in {0, 1, ..., B-1}
type SampleIndex = tuple[DemoIndex, TimeIndex]  # (i, t)

States: TypeAlias = list[State[DimState]]
Actions: TypeAlias = list[Action[DimAction]]
SampleIndices: TypeAlias = list[SampleIndex]

## Utils

SEED = 42
EPS: float = 1e-8
MAD_SCALE: float = 1.4826  # Gaussian consistency factor for MAD


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


set_seed(SEED)

if torch.backends.mps.is_available():
    torch_device_auto = torch.device("mps")
elif torch.cuda.is_available():
    torch_device_auto = torch.device("cuda")
else:
    torch_device_auto = torch.device("cpu")
console.print(f"Using device: [green]{torch_device_auto}[/green]")


def median(
    arr: npt.ArrayLike, /, axis: int | Sequence[int] | None = None
) -> np.ndarray:
    arr = np.asarray(arr)
    return np.median(arr, axis=axis)


def normalise(
    vec: onp.ToArray1D, /, method: Literal["NORM", "MINMAX", "ZSCORE"]
) -> np.ndarray:
    vec = np.asarray(vec, dtype=npDType)
    match method:
        case "NORM":
            norm = la.norm(vec)
            return vec / (norm + EPS)
        case "MINMAX" | "ZSCORE":
            min_: float = vec.min()
            max_: float = vec.max()
            return (vec - min_) / (max_ - min_ + EPS)


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

    # @enforce_shapes
    def states(self) -> States[DimState]:
        return list(sample.state for sample in self.samples)

    # @enforce_shapes
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
    def __getitem__(
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
            case int():
                return self.collection[index]

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

    # @enforce_shapes
    def states(self, *, LOO_demo_index: DemoIndex | None = None) -> States[DimState]:
        # (N x T_) or (N-1 x T_)
        return list(
            sample.state for sample in self.samples(LOO_demo_index=LOO_demo_index)
        )

    # @enforce_shapes
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
        self.n_pairs = len(self.states)  # T_i

    def __len__(self) -> int:
        return self.n_pairs

    @enforce_shapes
    def __getitem__(
        self, t: TimeIndex, /
    ) -> Sample[DimState, DimAction]:  # (x_{i, t}, a_{i, t})
        return Sample(self.states[t], self.actions[t])

    @enforce_shapes
    def __iter__(self) -> Iterator[Sample[DimState, DimAction]]:
        # [(x_{i, t}, a_{i, t})]_{t = 1}^{T_i}
        for t in range(self.n_pairs):
            yield self[t]

    @property
    def state_dim(self) -> DimState:  # d_x
        return cast(DimState, self.states[0].shape[0])

    @property
    def action_dim(self) -> DimAction:  # d_a
        return cast(DimAction, self.actions[0].shape[0])

    def sample(self, t: TimeIndex, /) -> Sample[DimState, DimAction]:
        return self[t]  # (x_{i, t}, a_{i, t})

    def samples(self) -> Samples[DimState, DimAction]:
        return Samples(list(sample for sample in self))

    @classmethod
    def from_samples(
        cls, index: DemoIndex, samples: Samples[DimState, DimAction]
    ) -> Self:
        states = list(sample.state for sample in samples)
        actions = list(sample.action for sample in samples)
        return cls(index=index, states=states, actions=actions)


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
    def __getitem__(self, index: SampleIndex) -> Sample[DimState, DimAction]: ...
    #
    def __getitem__(
        self, index: DemoIndex | SampleIndex
    ) -> Demonstration[DimState, DimAction] | Sample[DimState, DimAction]:
        match index:
            case tuple():
                i, t = index
                return self.demos[i][t]
            case int():
                return self.demos[index]

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

    @classmethod
    def from_samples_collection(
        cls, collection: SamplesCollection[DimState, DimAction]
    ) -> Self:
        demos = list[Demonstration[DimState, DimAction]]()
        for i, samples in enumerate(collection):
            demo = Demonstration[DimState, DimAction].from_samples(i, samples)
            demos.append(demo)
        return cls(demos=demos)


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


class PhaseEstimator(Generic[DimState, DimAction]):
    def __init__(
        self,
        demonstrations: Demonstrations[DimState, DimAction],
        *,
        hidden_dim: int = 128,
        margin: float = 1.0,
        lr: float = 1e-3,
        epochs: int = 240,
        device: torch.device = torch_device_auto,
    ) -> None:
        self.demonstrations = demonstrations
        self.margin = margin
        self.lr = lr
        self.epochs = epochs
        self.device = device

        state_dim = self.demonstrations.state_dim
        scorer = PhaseScorer(state_dim=state_dim, hidden_dim=hidden_dim)
        self.scorer = scorer.to(self.device)
        self.optimiser = torch.optim.Adam(self.scorer.parameters(), lr=self.lr)

    def compute_ranking_loss(self) -> Tensor:  # L_rank
        ranking_loss = torch.tensor(0.0, device=self.device)
        for demo in self.demonstrations:
            states = Tensor(np.array(demo.states)).float().to(self.device)
            scores: Tensor = self.scorer(states)  # (T_i,)
            diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (T_i, T_i)
            mask = torch.triu(torch.ones_like(diff), diagonal=1)
            loss_matrix = F.softplus(self.margin - diff) * mask
            ranking_loss += loss_matrix.mean()
        return ranking_loss

    def train(self) -> Tensor | None:
        self.scorer.train()
        loss = None
        for _epoch in range(self.epochs):
            self.optimiser.zero_grad()
            loss = self.compute_ranking_loss()
            loss.backward()  # type: ignore
            torch.nn.utils.clip_grad_norm_(self.scorer.parameters(), 1.0)
            self.optimiser.step()  # type: ignore
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
    sample_indices: SampleIndices = field(default_factory=SampleIndices)  # I_b
    samples_collection: SamplesCollection[DimState, DimAction] = field(
        default_factory=SamplesCollection[DimState, DimAction]
    )
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
class BinHandler(Generic[DimState, DimAction]):
    phase_estimator: PhaseEstimator[DimState, DimAction]
    n_bins: int = field(default=96, kw_only=True)  # B
    bins: list[Bin[DimState, DimAction]] = field(init=False)

    @property
    def demonstrations(self) -> Demonstrations[DimState, DimAction]:
        return self.phase_estimator.demonstrations

    def phase_range(self, bin_idx: BinIndex) -> tuple[Phase, Phase]:
        return (bin_idx / self.n_bins, (bin_idx + 1) / self.n_bins)

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
                tau: Phase = phases[i][t]
                bin_idx: BinIndex = min(int(tau * self.n_bins), self.n_bins - 1)
                assert bin_idx < self.n_bins
                bin = self.bins[bin_idx]
                sample_idx: SampleIndex = (i, t)
                bin.sample_indices.append(sample_idx)
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

    def LOO_sample_indices(
        self,
        bin_idx: BinIndex,  # b
        demo_idx: int,  # i
    ) -> SampleIndices:  # I_b^{(-i)}
        bin = self.bins[bin_idx]
        sample_indices = SampleIndices()
        for sample_idx in bin.sample_indices:
            i, _t = sample_idx
            if i == demo_idx:
                continue
            sample_indices.append(sample_idx)
        return sample_indices

    @deprecated
    def LOO_split_samples(
        self,
        bin_idx: BinIndex,  # b
        demo_idx: int,  # i
    ) -> tuple[Samples[DimState, DimAction], Samples[DimState, DimAction]]:
        bin = self.bins[bin_idx]
        sample_indices = self.LOO_sample_indices(bin_idx, demo_idx)
        demo_indices = list(i for i, _t in sample_indices)
        loo_samples = Samples[DimState, DimAction]()
        demo_samples = Samples[DimState, DimAction]()
        for i, sample in enumerate(bin.samples()):
            if i not in demo_indices:
                loo_samples.append(sample)
            else:
                demo_samples.append(sample)
        return loo_samples, demo_samples

    @enforce_shapes
    def compute_z_scores(self) -> list[list[npDType]]:  # (N x T_)
        N = len(self.demonstrations)

        # (N x N x T_)
        # [[[r^{(-j)}_{i, t}]_{t = 1}^{T_i}]_{i = 1}^{N}]_{j = 1}^{N}
        _action_residuals = [[list[npDType]() for _ in range(N)] for _ in range(N)]

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
                    residual = la.norm(action - bin_median_action)  # r^{-i}_{i, t}
                    self_action_residuals[j].append(residual)

                bin_action_residuals = list[npDType]()  # LOO
                for action in bin.actions(LOO_demo_index=j):
                    residual = la.norm(action - bin_median_action)  # r^{-j}_{i, t}
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
    ) -> list[list[npDType]]:  # (N x T_)
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
        bin_median_actions = list[Action[DimAction]]()
        bin_median_states = list[State[DimState]]()

        for bin in self.bins:
            stats = self.compute_robust_consensus_statistics(bin.samples())

            bin_median_action = stats.median_action  # alpha_a[b]
            bin_action_residuals = list[npDType]()
            for action in bin.actions():
                residual = la.norm(action - bin_median_action)  # r_{i, t}
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

    # @enforce_shapes
    def compute_pseudo_labels(
        self,
        trust_values: list[list[npDType]],
        *,
        debias_weight: npDType | float,  # lambda_{debias}
        sideways_attenuation_shrinkage: npDType | float = 0.5,  # rho_0
        speed_regularisation_influence: npDType | float = 0.5,  # eta_0
        temporal_smoothing_weight: npDType | float = 0.0,  # kappa
    ) -> list[Actions[DimAction]]:  # (N x T_)
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
                        s * (y2 / la.norm(y2) + EPS)
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
                ystar = Action[DimAction]((1 - kappa) * y3 + kappa * ystar_prev)
                pseudo_labels[i].append(ystar)
                ystar_prev = ystar

        return pseudo_labels


class BCPolicy(nn.Module, Generic[DimState, DimAction]):
    """Behavioral cloning policy that maps states to actions"""

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
        return self.network(states)


class PACER(Generic[DimState, DimAction]):
    def __init__(
        self,
        demonstrations: Demonstrations[DimState, DimAction],
        *,
        lr: float = 1e-3,
        epochs: int = 240,
        device: torch.device = torch_device_auto,
    ) -> None:
        self.demonstrations = demonstrations
        self.lr = lr
        self.epochs = epochs
        self.device = device

        policy = BCPolicy(
            state_dim=self.demonstrations.state_dim,
            action_dim=self.demonstrations.action_dim,
            hidden_dim=128,
        )
        self.policy = policy.to(self.device)
        self.optimiser = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def prepare(self) -> None:
        self.phase_estimator = PhaseEstimator(
            self.demonstrations,
            hidden_dim=128,
            margin=1,
            lr=1e-3,
            epochs=240,
            device=self.device,
        )
        self.phase_estimator.train()
        self.binner = BinHandler(self.phase_estimator, n_bins=96)
        self.binner.make_bins()
        self.trust_values = self.binner.compute_trust_values(
            cutoff=4.685,
            min_trust=0.02,
        )
        self.pseudo_labels = self.binner.compute_pseudo_labels(
            self.trust_values,
            debias_weight=0.5,
            sideways_attenuation_shrinkage=0.5,
            speed_regularisation_influence=0.5,
            temporal_smoothing_weight=0.0,
        )

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
            preds = self.policy(states)  # (T_i, action_dim)

            diffs = preds - targets  # (T_i, action_dim)
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

    def train(self) -> Tensor | None:
        """Train PACER policy using weighted Huber loss with pseudo-labels."""
        self.policy.train()
        loss = None
        for _epoch in range(self.epochs):
            self.optimiser.zero_grad()
            loss = self.compute_huber_loss()
            loss.backward()  # type: ignore
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimiser.step()  # type: ignore
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
