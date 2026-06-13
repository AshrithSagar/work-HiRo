"""
PACER Base
=======
Additional core structures for PACER.
"""
# src/pacer/pacer/base.py

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import TypeAlias

from typingkit.core import TypedList

from pacer.typings import NumDemos, NumPoints, npDType

## ── PACER ────────────────────────────────────────────────────────────────────

Residual: TypeAlias = npDType  # r_{i, t}
Residuals: TypeAlias = TypedList[NumPoints, Residual]
ResidualsCollection: TypeAlias = TypedList[NumDemos, Residuals[NumPoints]]

MetricValue: TypeAlias = npDType
MetricSeries: TypeAlias = TypedList[NumPoints, MetricValue]
MetricCollection: TypeAlias = TypedList[NumDemos, MetricSeries[NumPoints]]

## ─────────────────────────────────────────────────────────────────────────────
