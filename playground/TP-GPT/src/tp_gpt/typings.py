"""
Typing utils
=======
src/tp_gpt/typings.py
"""

from typing import Generic, TypeAlias, TypeVar

from typed_numpy._typed import DimVar, DimVarBinder, ShapedNDArray, TypedNDArray
from typed_numpy._typed.shapes import THREE, TWO

DimNT = TypeVar("DimNT", bound=int, default=int)
"""TypeVar denoting number of points"""
DimDT = TypeVar("DimDT", bound=int, default=int)
"""TypeVar denoting dimension of the space"""

Point: TypeAlias = TypedNDArray[tuple[DimDT]]
PointsArray: TypeAlias = TypedNDArray[tuple[DimNT, DimDT]]
RotationMatrix: TypeAlias = TypedNDArray[tuple[DimDT, DimDT]]
Jacobian: TypeAlias = TypedNDArray[tuple[DimDT, DimDT]]
JacobianArray: TypeAlias = TypedNDArray[tuple[DimNT, DimDT, DimDT]]


class Space(Generic[DimNT, DimDT], DimVarBinder):
    _N = DimVar()
    _D = DimVar()

    Point: ShapedNDArray[tuple[DimDT]] = ShapedNDArray(_D)
    PointsArray: ShapedNDArray[tuple[DimNT, DimDT]] = ShapedNDArray(_N, _D)
    RotationMatrix: ShapedNDArray[tuple[DimDT, DimDT]] = ShapedNDArray(_D, _D)
    Jacobian: ShapedNDArray[tuple[DimDT, DimDT]] = ShapedNDArray(_D, _D)
    JacobianArray: ShapedNDArray[tuple[DimNT, DimDT, DimDT]] = ShapedNDArray(_N, _D, _D)


class Space2D(Space[DimNT, TWO]): ...


class Space3D(Space[DimNT, THREE]): ...
