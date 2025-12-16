"""
Typing utils
=======
src/tp_gpt/typings.py
"""

from typing import TypeAlias

from typed_numpy import TypedNDArray
from typed_numpy.dimensions import DimT, GenericDim
from typed_numpy.shapes import THREE, TWO, Shape

Point: TypeAlias = TypedNDArray[tuple[DimT]]
PointsArray: TypeAlias = TypedNDArray[tuple[int, DimT]]
RotationMatrix: TypeAlias = TypedNDArray[tuple[DimT, DimT]]
Jacobian: TypeAlias = TypedNDArray[tuple[DimT, DimT]]
JacobianArray: TypeAlias = TypedNDArray[tuple[int, DimT, DimT]]


class Space(GenericDim[DimT]):
    Point: Shape[tuple[DimT]] = Shape("D")
    PointsArray: Shape[tuple[int, DimT]] = Shape("N", "D")
    RotationMatrix: Shape[tuple[DimT, DimT]] = Shape("D", "D")
    Jacobian: Shape[tuple[DimT, DimT]] = Shape("D", "D")
    JacobianArray: Shape[tuple[int, DimT, DimT]] = Shape("N", "D", "D")


class Space2D(Space[TWO]): ...


class Space3D(Space[THREE]): ...
