"""
PACER Base
=======
Additional core structures for PACER.
"""
# src/pacer/pacer/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Self, TypeAlias

from typingkit.core import TypedDict, TypedList

from pacer.base import Demonstration, Demonstrations
from pacer.typings import DemoIndex, DimAction, DimState, NumDemos, NumPoints, npDType

## ── PACER ────────────────────────────────────────────────────────────────────

Residual: TypeAlias = npDType  # r_{i, t}
Residuals: TypeAlias = TypedList[NumPoints, Residual]
ResidualsCollection: TypeAlias = TypedList[NumDemos, Residuals[NumPoints]]

MetricValue: TypeAlias = npDType
MetricSeries: TypeAlias = TypedList[NumPoints, MetricValue]
MetricCollection: TypeAlias = TypedList[NumDemos, MetricSeries[NumPoints]]

# ──────────────────────────────────────────────────────────────────────────────

ZScore: TypeAlias = npDType  # z_{i, t}
"""Normalised residual (z-score)."""


class ZScores(TypedList[NumPoints, ZScore]):
    @classmethod
    def zeros_like(cls, demo: Demonstration[NumPoints, DimState, DimAction]) -> Self:
        T_i = demo.time_indices.length
        return cls.full(T_i, ZScore(0))


class ZScoresCollection(TypedDict[NumDemos, DemoIndex, ZScores[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demos.demo_indices, lambda i: ZScores[NumPoints].zeros_like(demos[i])
        )


# ──────────────────────────────────────────────────────────────────────────────

TrustValue: TypeAlias = npDType  # w_{i, t}
"""Confidence weight for a sample."""


class TrustValues(TypedList[NumPoints, TrustValue]):
    @classmethod
    def zeros_like(cls, demo: Demonstration[NumPoints, DimState, DimAction]) -> Self:
        T_i = demo.time_indices.length
        return cls.full(T_i, TrustValue(0))


class TrustValuesCollection(TypedDict[NumDemos, DemoIndex, TrustValues[NumPoints]]):
    @classmethod
    def zeros_like(
        cls, demos: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    ) -> Self:
        return cls.full(
            demos.demo_indices, lambda i: TrustValues[NumPoints].zeros_like(demos[i])
        )


## ─────────────────────────────────────────────────────────────────────────────
