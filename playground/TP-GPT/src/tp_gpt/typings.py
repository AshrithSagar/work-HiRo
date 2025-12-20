"""
Typing utils
=======
src/tp_gpt/typings.py
"""

from typing import Generic, TypeAlias, TypeVar

from typed_numpy._typed import DimVar, DimVarBinder, TypedNDArray
from typed_numpy._typed import ShapedNDArray as Shaped
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

    _Point: Shaped[tuple[DimDT]] = Shaped(_D)
    _PointsArray: Shaped[tuple[DimNT, DimDT]] = Shaped(_N, _D)
    _RotationMatrix: Shaped[tuple[DimDT, DimDT]] = Shaped(_D, _D)
    _Jacobian: Shaped[tuple[DimDT, DimDT]] = Shaped(_D, _D)
    _JacobianArray: Shaped[tuple[DimNT, DimDT, DimDT]] = Shaped(_N, _D, _D)


class Space2D(Space[DimNT, TWO]): ...


class Space3D(Space[DimNT, THREE]): ...
