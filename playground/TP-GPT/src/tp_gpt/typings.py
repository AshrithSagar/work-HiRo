"""
Typing utils
=======
src/tp_gpt/typings.py
"""

from typing import TypeAlias, TypeVar

from typed_numpy import TypedNDArray
from typed_numpy.shapes import THREE, TWO

TwoD: TypeAlias = TWO
ThreeD: TypeAlias = THREE

DimT = TypeVar("DimT", TwoD, ThreeD)


Point: TypeAlias = TypedNDArray[tuple[DimT]]
PointsArray: TypeAlias = TypedNDArray[tuple[int, DimT]]
RotationMatrix: TypeAlias = TypedNDArray[tuple[DimT, DimT]]
Jacobian: TypeAlias = TypedNDArray[tuple[DimT, DimT]]
JacobianArray: TypeAlias = TypedNDArray[tuple[int, DimT, DimT]]
