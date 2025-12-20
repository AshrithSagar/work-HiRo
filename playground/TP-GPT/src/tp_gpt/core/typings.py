"""
Typing utils
=======
src/tp_gpt/typings.py
"""

from typing import Generic, TypeAlias, TypeVar

from typed_numpy._typed import DimVar
from typed_numpy._typed import DimVarBinder as Dimensioned
from typed_numpy._typed import ShapedNDArray as Shaped
from typed_numpy._typed import TypedNDArray as NDArray
from typed_numpy._typed.shapes import THREE, TWO

## Static bindings

TwoD: TypeAlias = TWO
ThreeD: TypeAlias = THREE


DimSpace = TypeVar("DimSpace", bound=int, default=int)
"""TypeVar denoting dimension of the space"""

Point: TypeAlias = NDArray[tuple[DimSpace]]
Vector: TypeAlias = NDArray[tuple[DimSpace]]
RotationMatrix: TypeAlias = NDArray[tuple[DimSpace, DimSpace]]


NumPoints = TypeVar("NumPoints", bound=int, default=int)
"""TypeVar denoting number of points"""

ScalarArray: TypeAlias = NDArray[tuple[NumPoints]]
PointSet: TypeAlias = NDArray[tuple[NumPoints, DimSpace]]
JacobianArray: TypeAlias = NDArray[tuple[NumPoints, DimSpace, DimSpace]]


## Runtime bindings

_DimSpace = DimVar()
"""DimVar denoting dimension of the space"""

_NumPoints = DimVar()
"""DimVar denoting number of points"""


class CoordinateSpace(Generic[DimSpace], Dimensioned):
    _DimSpace = _DimSpace

    _Point: Shaped[tuple[DimSpace]] = Shaped(_DimSpace)
    _Vector: Shaped[tuple[DimSpace]] = Shaped(_DimSpace)
    _RotationMatrix: Shaped[tuple[DimSpace, DimSpace]] = Shaped(_DimSpace, _DimSpace)


class CoordinateSpace2D(CoordinateSpace[TwoD]): ...


class CoordinateSpace3D(CoordinateSpace[ThreeD]): ...


class PointSetSpace(Generic[NumPoints, DimSpace], CoordinateSpace[DimSpace]):
    _NumPoints = _NumPoints
    _DimSpace = _DimSpace

    _ScalarArray: Shaped[tuple[NumPoints]] = Shaped(_NumPoints)
    _PointSet: Shaped[tuple[NumPoints, DimSpace]] = Shaped(_NumPoints, _DimSpace)
    _JacobianArray: Shaped[tuple[NumPoints, DimSpace, DimSpace]] = Shaped(
        _NumPoints, _DimSpace, _DimSpace
    )


class PointSetSpace2D(PointSetSpace[NumPoints, TwoD]): ...


class PointSetSpace3D(PointSetSpace[NumPoints, ThreeD]): ...
