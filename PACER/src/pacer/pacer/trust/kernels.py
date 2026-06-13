"""
Trust Kernels
=======
"""
# src/pacer/pacer/trust/kernels.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pacer.pacer.trust.base import TrustValue, ZScore
from pacer.typings import FloatLike

## ── Trust Kernels ────────────────────────────────────────────────────────────


class TrustKernel(Protocol):
    """Maps robust z-score to trust value."""

    def compute(self, z_score: ZScore) -> TrustValue: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TukeyBiweightKernel:
    """Tukey biweight robust trust kernel."""

    cutoff: FloatLike = 4.685  # c

    def __post_init__(self) -> None:
        assert 3 <= self.cutoff <= 5

    def compute(self, z_score: ZScore) -> TrustValue:
        weight: FloatLike
        if z_score <= self.cutoff:
            weight = (1 - (z_score / self.cutoff) ** 2) ** 2
        else:
            weight = 0.0
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class GaussianKernel:
    sigma: FloatLike = 1.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = float(np.exp(-0.5 * (z_score / self.sigma) ** 2))
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class HuberKernel:
    """Huber weighting function."""

    delta: FloatLike = 1.345
    # 95% efficiency under Gaussian

    def compute(self, z_score: ZScore) -> TrustValue:
        weight: FloatLike
        abs_z_score = abs(z_score)
        if abs_z_score <= self.delta:
            weight = 1.0
        else:
            weight = self.delta / abs_z_score
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class WelschKernel:
    """Welsch (exponential) weighting function."""

    scale: FloatLike = 2.9846  # Tuning constant

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = np.exp(-((z_score / self.scale) ** 2))
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class CauchyKernel:
    """Cauchy (Lorentzian) weighting function."""

    scale: FloatLike = 2.3849  # Tuning constant

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = 1.0 / (1.0 + (z_score / self.scale) ** 2)
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class LogisticKernel:
    midpoint: FloatLike = 2.0
    sharpness: FloatLike = 2.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = 1.0 / (1.0 + np.exp(self.sharpness * (z_score - self.midpoint)))
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class HardThresholdKernel:
    cutoff: FloatLike = 3.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = 1.0 if z_score <= self.cutoff else 0.0
        return TrustValue(weight)


@dataclass(frozen=True, kw_only=True, slots=True)
class StudentTKernel:
    dof: FloatLike = 4.0

    def compute(self, z_score: ZScore) -> TrustValue:
        weight = (self.dof + 1) / (self.dof + z_score**2)
        return TrustValue(min(1.0, weight))


@dataclass(frozen=True, kw_only=True, slots=True)
class GemanMcClureKernel:
    """Geman-McClure weighting function."""

    sigma: FloatLike = 1.0

    def compute(self, z_score: ZScore) -> TrustValue:
        denom = 1.0 + (z_score / self.sigma) ** 2
        return TrustValue(1.0 / (denom * denom))


@dataclass(frozen=True, kw_only=True, slots=True)
class AndrewsSineKernel:
    """Andrews sine weighting function."""

    k: FloatLike = 1.34 * np.pi  # ~4.21

    def compute(self, z_score: ZScore) -> TrustValue:
        weight: FloatLike
        if abs(z_score) < self.k:
            t = z_score / self.k
            weight = np.sin(t) / t if t != 0 else 1.0
        else:
            weight = 0.0
        return TrustValue(weight)


# ── Trust Transforms ──────────────────────────────────────────────────────────


class TrustTransform(Protocol):
    """Post-processing transform on trust values."""

    def apply(self, trust: TrustValue) -> TrustValue: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class MinimumTrustFloor:
    """Applies minimum trust floor."""

    minimum: FloatLike = 0.02  # w_min

    def apply(self, trust: TrustValue) -> TrustValue:
        return TrustValue(max(trust, self.minimum))


## ─────────────────────────────────────────────────────────────────────────────
