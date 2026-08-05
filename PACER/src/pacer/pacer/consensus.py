"""
Consensus
=========
Consensus estimators for PACER.
"""
# src/pacer/pacer/consensus.py

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, override

import numpy as np
import numpy.typing as npt
import scipy
from typingkit.numpy._typed.helpers import Dim1

from pacer.base import Action, Actions, State, States
from pacer.pacer.base import MetricValue, Residual
from pacer.typings import DimAction, DimState, Matrix, NumPoints, Vector, VectorsType
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


class WindowParametrisation(Protocol):
    """
    Computes a scalar parameter associated with each point,
    used as the independent variable for local tangent estimation.
    """

    def compute(self, vectors: VectorsType) -> Vector[Any]: ...


@dataclass(frozen=True, slots=True)
class IndexParametrisation(WindowParametrisation):
    """
    Parametrises by raw point index.
    Assumes uniform spacing between points.
    """

    @override
    def compute(self, vectors: VectorsType) -> Vector[Any]:
        return Vector(np.arange(len(vectors)))


@dataclass(frozen=True, slots=True)
class ArcLengthParametrisation(WindowParametrisation):
    """
    Parametrises by cumulative Euclidean arc-length.
    Corrects for uneven point spacing along the trajectory.
    """

    @override
    def compute(self, vectors: VectorsType) -> Vector[Any]:
        ds = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
        return Vector(np.concatenate([[0.0], np.cumsum(ds)]))


class LocalLineFitter(Protocol):
    """Fits a local linear model over a parameterised window and returns the estimated tangent (slope)."""

    min_points: int
    """
    Minimum number of samples required for a valid fit.
    Smaller windows fall back to a finite-difference estimate.
    """

    def fit_slope(
        self, x: Vector[NumPoints], y: Matrix[NumPoints, Dim1]
    ) -> Vector[Dim1]: ...


@dataclass(frozen=True, slots=True)
class OLSLineFitter(LocalLineFitter):
    """Ordinary-least-squares line fit. Sensitive to outliers within the window."""

    min_points: int = 2

    @override
    def fit_slope(
        self, x: Vector[NumPoints], y: Matrix[NumPoints, Dim1]
    ) -> Vector[Dim1]:
        return Vector(np.polyfit(x, y, deg=1)[0])


@dataclass(frozen=True, slots=True)
class HuberLineFitter(LocalLineFitter):
    """Robust (Huber) line fit. Down-weights outlier samples within the window."""

    epsilon: float = 1.35  # Huber threshold
    min_points: int = 3

    @override
    def fit_slope(
        self, x: Vector[NumPoints], y: Matrix[NumPoints, Dim1]
    ) -> Vector[Dim1]:
        from sklearn.linear_model import HuberRegressor

        dim = y.shape[1]
        slope = np.empty(dim)
        x_col = x.reshape(-1, 1)
        for d in range(dim):
            try:
                reg = HuberRegressor(epsilon=self.epsilon).fit(x_col, y[:, d])
                slope[d] = reg.coef_[0]
            except ValueError:
                # Degenerate fit (e.g. constant values); fall back to OLS.
                slope[d] = np.polyfit(x, y[:, d], deg=1)[0]
        return Vector(slope)


@dataclass(frozen=True)
class LocalWindowTangentEstimator(TangentEstimator):
    """Estimates tangents by fitting a local linear model over a sliding window centred at each sample."""

    window_length: int = 5  # Points in the fitting window (odd, >= 3)
    parametrisation: WindowParametrisation = field(default_factory=IndexParametrisation)
    fitter: LocalLineFitter = field(default_factory=OLSLineFitter)

    def __post_init__(self) -> None:
        assert self.window_length >= 3
        assert self.window_length % 2 == 1

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        n = len(vectors)
        half = self.window_length // 2
        param = self.parametrisation.compute(vectors)
        tangents = np.empty_like(vectors)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            if hi - lo < self.fitter.min_points:
                # Window too small for the chosen fitter; use a local finite difference instead.
                j, k = max(i - 1, 0), min(i + 1, n - 1)
                denom = param[k] - param[j]
                tangents[i] = (
                    (vectors[k] - vectors[j]) / denom if abs(denom) > EPS else 0.0
                )
                continue
            # Centre parameter values on the current sample.
            x = Vector(param[lo:hi] - param[i])
            if np.ptp(x) < EPS:  # Degenerate parameter window.
                tangents[i] = 0.0
                continue
            y = Matrix(vectors[lo:hi])
            tangents[i] = self.fitter.fit_slope(x, y)
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class ArcLengthTangentEstimator(TangentEstimator):
    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        ds = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        tangents = np.asarray(np.gradient(vectors, s, axis=0))
        return vectors.from_array(tangents)


@dataclass(frozen=True)
class ArcLengthLocalLinearTangentEstimator(TangentEstimator):
    window_length: int = 5

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        return LocalWindowTangentEstimator(
            window_length=self.window_length,
            parametrisation=ArcLengthParametrisation(),
            fitter=OLSLineFitter(),
        ).compute(vectors)


@dataclass(frozen=True)
class LocalLinearTangentEstimator(TangentEstimator):
    """
    Estimates tangents via an ordinary-least-squares line
    fit over a local sliding window centred on each bin.
    """

    window_length: int = 5

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        return LocalWindowTangentEstimator(
            window_length=self.window_length,
            parametrisation=IndexParametrisation(),
            fitter=OLSLineFitter(),
        ).compute(vectors)


@dataclass(frozen=True)
class RobustLocalLinearTangentEstimator(TangentEstimator):
    """
    Estimates tangents via a robust (Huber) line fit
    over a local sliding window centred on each bin.
    Down-weights outlier samples within the window.
    """

    window_length: int = 5
    epsilon: float = 1.35

    @override
    def compute(self, vectors: VectorsType) -> VectorsType:
        return LocalWindowTangentEstimator(
            window_length=self.window_length,
            parametrisation=IndexParametrisation(),
            fitter=HuberLineFitter(epsilon=self.epsilon),
        ).compute(vectors)


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    vector_estimator: VectorLocationEstimator = MedianVectorEstimator()
    scalar_estimator: ScalarLocationEstimator = MedianScalarEstimator()
    residual_scale_estimator: ResidualScaleEstimator = MADResidualScaleEstimator()
    tangent_estimator: TangentEstimator = CentralDifferenceTangentEstimator()


## ─────────────────────────────────────────────────────────────────────────────
