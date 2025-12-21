"""
Space abstractions
=======
src/tp_gpt/core/spaces.py
"""

from typing import Generic, TypeAlias

from typed_numpy._typed import DimVarBinder as Dimensioned
from typed_numpy._typed import ShapedNDArray as Shaped
from typed_numpy._typed import TypedNDArray as NDArray

from tp_gpt.core.typings import DimSpace, NumPoints, _DimSpace, _NumPoints

Point: TypeAlias = NDArray[tuple[DimSpace]]
Vector: TypeAlias = NDArray[tuple[DimSpace]]
RotationMatrix: TypeAlias = NDArray[tuple[DimSpace, DimSpace]]

ScalarArray: TypeAlias = NDArray[tuple[NumPoints]]
PointSet: TypeAlias = NDArray[tuple[NumPoints, DimSpace]]


class CoordinateSpace(Generic[DimSpace], Dimensioned):
    _DimSpace = _DimSpace

    _Point: Shaped[tuple[DimSpace]] = Shaped(_DimSpace)
    _Vector: Shaped[tuple[DimSpace]] = Shaped(_DimSpace)
    _RotationMatrix: Shaped[tuple[DimSpace, DimSpace]] = Shaped(_DimSpace, _DimSpace)


class SpaceCollection(Generic[NumPoints, DimSpace], CoordinateSpace[DimSpace]):
    _NumPoints = _NumPoints
    _DimSpace = _DimSpace

    _ScalarArray: Shaped[tuple[NumPoints]] = Shaped(_NumPoints)
    _PointSet: Shaped[tuple[NumPoints, DimSpace]] = Shaped(_NumPoints, _DimSpace)
