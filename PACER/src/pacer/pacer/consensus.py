"""
Consensus
=========
Consensus estimators for PACER.
"""
# src/pacer/pacer/consensus.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, override

import numpy as np
import numpy.typing as npt
import scipy

from pacer.base import Action, Actions, State, States
from pacer.pacer.base import MetricValue, Residual
from pacer.typings import DimAction, DimState, NumPoints, VectorsType
from pacer.utils import EPS, mean, median

## ── Consensus ────────────────────────────────────────────────────────────────

# ── Location Estimation ───────────────────────────────────────────────────────


class VectorLocationEstimator(Protocol):
    """Computes consensus vector estimate."""

    def compute_action(
        self, actions: Actions[NumPoints, DimAction]
    ) -> Action[DimAction]: ...

    def compute_state(self, states: States[NumPoints, DimState]) -> State[DimState]: ...


class ScalarLocationEstimator(Protocol):
    """Computes scalar consensus estimate."""

    def compute(self, values: npt.ArrayLike) -> MetricValue: ...


# ── Mean Estimators ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MeanVectorEstimator:
    def compute_action(
        self, actions: Actions[NumPoints, DimAction]
    ) -> Action[DimAction]:
        return Action[DimAction](mean(actions, axis=0))

    def compute_state(self, states: States[NumPoints, DimState]) -> State[DimState]:
        return State[DimState](mean(states, axis=0))


@dataclass(frozen=True, slots=True)
class MeanScalarEstimator:
    def compute(self, values: npt.ArrayLike) -> MetricValue:
        return MetricValue(mean(values))


# ── Median Estimators ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MedianVectorEstimator:
    def compute_action(
        self, actions: Actions[NumPoints, DimAction]
    ) -> Action[DimAction]:
        return Action[DimAction](median(actions, axis=0))

    def compute_state(self, states: States[NumPoints, DimState]) -> State[DimState]:
        return State[DimState](median(states, axis=0))


@dataclass(frozen=True, slots=True)
class MedianScalarEstimator:
    def compute(self, values: npt.ArrayLike) -> MetricValue:
        return MetricValue(median(values))


# ── Scale Estimation ──────────────────────────────────────────────────────────


class ResidualScaleEstimator(Protocol):
    """Computes robust scale from residuals."""

    def compute(self, residuals: Sequence[Residual]) -> Residual: ...


@dataclass(frozen=True, slots=True)
class MADResidualScaleEstimator:
    """Median absolute deviation scale estimator."""

    scale: float | Literal["normal"] = "normal"
    """
    Consistency scale.
    The numerical value of scale will be divided out of the final result.\\
    scale="normal" => scale=scipy.special.ndtri(0.75)
    => Gaussian consistency factor for MAD ~= 1/0.67449 ~= 1.4826
    """

    def compute(self, residuals: Sequence[Residual]) -> Residual:
        return Residual(scipy.stats.median_abs_deviation(residuals, scale=self.scale))


@dataclass(frozen=True, slots=True)
class StandardDeviationScaleEstimator:
    ddof: int = 0

    def compute(self, residuals: Sequence[Residual]) -> Residual:
        return Residual(np.std(residuals, ddof=self.ddof))


# ── Tangent Estimation ────────────────────────────────────────────────────────


class TangentEstimator(Protocol):
    def compute(self, vectors: VectorsType) -> VectorsType: ...


@dataclass(frozen=True)
class IdentityTangentEstimator(TangentEstimator):
    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        return vectors


@dataclass(frozen=True)
class CentralDifferenceTangentEstimator(TangentEstimator):
    edge_order: Literal[1, 2] = 2

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        tangents = np.gradient(vectors, axis=0, edge_order=self.edge_order)
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class ForwardDifferenceTangentEstimator(TangentEstimator):
    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        arr = vectors.numpy()
        tangents = np.empty_like(arr)
        tangents[:-1] = arr[1:] - arr[:-1]
        tangents[-1] = tangents[-2]
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class UnitTangentEstimator(TangentEstimator):
    epsilon: float = EPS

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        tangents = np.gradient(vectors, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / np.maximum(norms, self.epsilon)
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class ArcLengthTangentEstimator(TangentEstimator):
    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        arr = vectors.numpy()
        ds = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        tangents = np.asarray(np.gradient(arr, s, axis=0))
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class GaussianTangentEstimator(TangentEstimator):
    sigma: float = 1.0

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        arr = vectors.numpy()
        smooth = scipy.ndimage.gaussian_filter1d(
            arr, sigma=self.sigma, axis=0, mode="nearest"
        )
        tangents = np.asarray(np.gradient(smooth, axis=0))
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class SavitzkyGolayTangentEstimator(TangentEstimator):
    window_length: int = 7
    polyorder: int = 3

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        tangents = scipy.signal.savgol_filter(
            vectors,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=1,
            axis=0,
            mode="interp",
        )
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class SavitzkyGolaySmoothedTangentEstimator(TangentEstimator):
    window_length: int = 7
    polyorder: int = 3

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        smooth = scipy.signal.savgol_filter(
            vectors,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=0,
            axis=0,
            mode="interp",
        )
        tangents = np.gradient(smooth, axis=0)
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class LocalLinearTangentEstimator(TangentEstimator):
    """
    Estimates tangents via an ordinary-least-squares line
    fit over a local sliding window centred on each bin.
    """

    window_length: int = 5  # Points in the fitting window (odd, >= 3)

    def __post_init__(self) -> None:
        assert self.window_length >= 3
        assert self.window_length % 2 == 1

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        arr = vectors.numpy()
        n = arr.shape[0]
        half = self.window_length // 2
        tangents = np.empty_like(arr)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            if hi - lo < 2:
                tangents[i] = 0.0
                continue
            x = np.arange(lo, hi, dtype=float) - i  # Centred index offsets
            y = arr[lo:hi]
            slope, _intercept = np.polyfit(x, y, deg=1)
            tangents[i] = slope
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class RobustLocalLinearTangentEstimator(TangentEstimator):
    """
    Estimates tangents via a robust (Huber) line fit
    over a local sliding window centred on each bin.
    Down-weights outlier samples within the window.
    """

    window_length: int = 5  # Points in the fitting window (odd, >= 3)
    epsilon: float = 1.35  # Huber threshold

    def __post_init__(self) -> None:
        assert self.window_length >= 3
        assert self.window_length % 2 == 1

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        from sklearn.linear_model import HuberRegressor

        arr = vectors.numpy()
        n, dim = arr.shape
        half = self.window_length // 2
        tangents = np.empty_like(arr)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            if hi - lo < 3:
                # Not enough points for a robust fit; fall back to a local finite-difference slope.
                j, k = max(i - 1, 0), min(i + 1, n - 1)
                tangents[i] = (arr[k] - arr[j]) / max(k - j, 1)
                continue
            x = (np.arange(lo, hi, dtype=float) - i).reshape(-1, 1)
            y = arr[lo:hi]
            for d in range(dim):
                try:
                    reg = HuberRegressor(epsilon=self.epsilon).fit(x, y[:, d])
                    tangents[i, d] = reg.coef_[0]
                except ValueError:
                    # Degenerate window (e.g. constant values); fall back to OLS.
                    tangents[i, d] = np.polyfit(x.ravel(), y[:, d], deg=1)[0]
        return vectors.from_array(tangents)


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    vector_estimator: VectorLocationEstimator = MedianVectorEstimator()
    scalar_estimator: ScalarLocationEstimator = MedianScalarEstimator()
    residual_scale_estimator: ResidualScaleEstimator = MADResidualScaleEstimator()
    tangent_estimator: TangentEstimator = CentralDifferenceTangentEstimator()


## ─────────────────────────────────────────────────────────────────────────────
