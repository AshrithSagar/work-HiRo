"""
Typings
=======
Typing utils
"""
# src/pacer/typings.py

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import NamedTuple, TypeAlias, TypeVar

import numpy as np
import torch
from typingkit.core import TypedList

## ── Typings ──────────────────────────────────────────────────────────────────

npDType: TypeAlias = np.float32
torchDType = torch.float32

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
NumDemos = TypeVar("NumDemos", bound=int, default=int)  # N
NumBins = TypeVar("NumBins", bound=int, default=int)  # B

# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────

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

## ─────────────────────────────────────────────────────────────────────────────
