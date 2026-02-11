"""
LASA Dataset
=======
"""
# src/pacer/lasa.py

from typing import Literal, TypeAlias, TypeVar

import numpy as np
import pyLasaDataset as lasa  # type: ignore
from pyLasaDataset.dataset import _Data  # type: ignore
from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.helpers import THREE, TWO

SEVEN = Literal[7]
THOUSAND = Literal[1000]

DType: TypeAlias = np.float32
Dim1 = TypeVar("Dim1", bound=int, default=int)
Dim2 = TypeVar("Dim2", bound=int, default=int)
Dim3 = TypeVar("Dim3", bound=int, default=int)
Array1D: TypeAlias = TypedNDArray[tuple[Dim1], np.dtype[DType]]
Array2D: TypeAlias = TypedNDArray[tuple[Dim1, Dim2], np.dtype[DType]]
Array3D: TypeAlias = TypedNDArray[tuple[Dim1, Dim2, Dim3], np.dtype[DType]]

NumPoints = TypeVar("NumPoints", bound=int, default=int)
Point2D: TypeAlias = Array1D[TWO]
Point3D: TypeAlias = Array1D[THREE]
Points2D: TypeAlias = Array2D[NumPoints, TWO]
Points3D: TypeAlias = Array2D[NumPoints, THREE]


class LASADemonstrations:
    def __init__(self, data: _Data = lasa.DataSet.GShape) -> None:
        self.data = data

        Ar7k2 = Array3D[SEVEN, THOUSAND, TWO]
        self.positions = Ar7k2([demo.__getattribute__("pos").T for demo in data.demos])
        self.velocities = Ar7k2([demo.__getattribute__("vel").T for demo in data.demos])
        self.positions_diff = np.diff(
            self.positions, axis=-2, append=np.zeros((7, 1, 2), dtype=DType)
        )
