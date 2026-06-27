"""
Typings
=======
Typing utils
"""
# src/pacer/typings.py

# pyright: reportPrivateImportUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Any, NamedTuple, Self, TypeAlias, TypeVar, cast

import numpy as np
import optype.numpy as onp
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

Matrix: TypeAlias = Array2D[Dim1, Dim2, np.dtype[npDType]]


class Vector(Array1D[Dim1, np.dtype[npDType]]):
    def __new__(cls, object: onp.ToArrayStrict1D) -> Self:
        return cast(Self, super().__new__(cls, object, dtype=npDType))

    @property
    def dim(self) -> Dim1:
        return self.shape[0]

    @classmethod
    def zeros(cls, dim: Dim1) -> Self:
        return cls(np.zeros((dim,)))


VectorType = TypeVar("VectorType", bound=Vector[Any], default=Vector)
# Ideally, we'd want HKTs; To bound to `Vector[Dim1]` instead of `Vector[Any]`.


class Vectors(TypedList[NumPoints, VectorType]):
    @property
    def dim(self) -> Any:
        return self[0].dim

    def numpy(self) -> Matrix[NumPoints, Any]:
        return Matrix[NumPoints, Any](self)

    def coord(self, dim: int) -> Vector[NumPoints]:
        return Vector[NumPoints](self.numpy()[:, dim])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Self:
        return cls(arr)


VectorsType = TypeVar("VectorsType", bound=Vectors[Any, Any], default=Vectors)

CollectionType = TypeVar("CollectionType")

# ── Indices ───────────────────────────────────────────────────────────────────

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
