"""
Typings
=======
Typing utils
"""
# src/pacer/typings.py

# pyright: reportPrivateImportUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import NamedTuple, TypeAlias, TypeVar

import numpy as np
import torch
from typingkit.core import TypedList
from typingkit.numpy._typed.helpers import Array1D, Array2D, Dim1, Dim2

## ── Typings ──────────────────────────────────────────────────────────────────

DimState = TypeVar("DimState", bound=int, default=int)  # d_x
"""State dimension `d_x`."""

DimAction = TypeVar("DimAction", bound=int, default=int)  # d_a
"""Action dimension `d_a`."""

NumPoints = TypeVar("NumPoints", bound=int, default=int)  # T_i
"""Number of time steps `T_i` in a demonstration `i`."""

NumDemos = TypeVar("NumDemos", bound=int, default=int)  # N
"""Number of demonstrations `N`."""

NumBins = TypeVar("NumBins", bound=int, default=int)  # B
"""Number of phase bins `B`."""

# ──────────────────────────────────────────────────────────────────────────────

FloatLike: TypeAlias = np.floating | float

npDType: TypeAlias = np.float32
"""The default NumPy scalar dtype."""

torchDType = torch.float32
"""The default PyTorch dtype."""

Vector: TypeAlias = Array1D[Dim1, np.dtype[npDType]]
Matrix: TypeAlias = Array2D[Dim1, Dim2, np.dtype[npDType]]

VectorType = TypeVar("VectorType", bound=Vector, default=Vector)
# Ideally, we'd want HKTs; To bound to `Vector[Dim1]` instead of `Vector[Any]`.

Vectors: TypeAlias = TypedList[NumPoints, VectorType]
VectorsType = TypeVar("VectorsType", bound=Vectors, default=Vectors)

CollectionType = TypeVar("CollectionType")

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
