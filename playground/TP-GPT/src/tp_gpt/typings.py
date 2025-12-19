"""
Typing utils
=======
src/tp_gpt/typings.py
"""

from typing import TypeAlias

from typed_numpy._typed import ShapedNDArray, TypedNDArray
from typed_numpy._typed.generics import DimT, DimVar, GenericDim
from typed_numpy._typed.shapes import THREE, TWO

Point: TypeAlias = TypedNDArray[tuple[DimT]]
PointsArray: TypeAlias = TypedNDArray[tuple[int, DimT]]
RotationMatrix: TypeAlias = TypedNDArray[tuple[DimT, DimT]]
Jacobian: TypeAlias = TypedNDArray[tuple[DimT, DimT]]
JacobianArray: TypeAlias = TypedNDArray[tuple[int, DimT, DimT]]


class Space(GenericDim[DimT]):
    D = DimVar()

    Point: ShapedNDArray[tuple[DimT]] = ShapedNDArray(D)
    PointsArray: ShapedNDArray[tuple[int, DimT]] = ShapedNDArray(None, D)
    RotationMatrix: ShapedNDArray[tuple[DimT, DimT]] = ShapedNDArray(D, D)
    Jacobian: ShapedNDArray[tuple[DimT, DimT]] = ShapedNDArray(D, D)
    JacobianArray: ShapedNDArray[tuple[int, DimT, DimT]] = ShapedNDArray(None, D, D)


class Space2D(Space[TWO]): ...


class Space3D(Space[THREE]): ...
