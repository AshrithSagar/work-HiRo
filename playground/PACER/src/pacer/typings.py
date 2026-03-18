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
from typingkit.numpy._typed.helpers import Array1D, Array2D, Dim1, Dim2

## ── Typings ──────────────────────────────────────────────────────────────────

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
NumDemos = TypeVar("NumDemos", bound=int, default=int)  # N
NumBins = TypeVar("NumBins", bound=int, default=int)  # B

# ──────────────────────────────────────────────────────────────────────────────

npDType: TypeAlias = np.float32
torchDType = torch.float32

Vector: TypeAlias = Array1D[Dim1, np.dtype[npDType]]
Matrix: TypeAlias = Array2D[Dim1, Dim2, np.dtype[npDType]]

# ──────────────────────────────────────────────────────────────────────────────

Phase: TypeAlias = npDType  # tau \in [0, 1]
Phases: TypeAlias = TypedList[NumPoints, Phase]
PhasesCollection: TypeAlias = TypedList[NumDemos, Phases[NumPoints]]

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
