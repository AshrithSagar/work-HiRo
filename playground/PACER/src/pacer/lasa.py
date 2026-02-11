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
from typed_numpy._typed.shapes import THREE, TWO

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
        positions = list[Points3D[THOUSAND]]()
        velocities = list[Points3D[THOUSAND]]()
        for demo in data.demos:
            pos = Array2D[TWO, THOUSAND](demo.__getattribute__("pos"))
            vel = Array2D[TWO, THOUSAND](demo.__getattribute__("vel"))
            n_points = pos.shape[1]  # T_i
            poss = Points3D[THOUSAND](
                [(pos[0, t], pos[1, t], 0.0) for t in range(n_points)]
            )
            vels = Points3D[THOUSAND](
                [(vel[0, t], vel[1, t], 0.0) for t in range(n_points)]
            )
            positions.append(poss)
            velocities.append(vels)
        self.positions = Array3D[SEVEN, THOUSAND, THREE](positions)
        self.velocities = Array3D[SEVEN, THOUSAND, THREE](velocities)
        self.positions_diff = np.diff(
            self.positions, axis=-2, append=np.zeros((7, 1, 3), dtype=DType)
        )
