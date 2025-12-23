"""
Typing utils
=======
src/tp_gpt/core/typings.py
"""

from typing import Any, Protocol, TypeAlias, TypeVar

from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.shapes import THREE, TWO

DimSpace = TypeVar("DimSpace", bound=int, default=int)
"""TypeVar denoting dimension of the space"""

NumPoints = TypeVar("NumPoints", bound=int, default=int)
"""TypeVar denoting number of points"""

TwoD: TypeAlias = TWO
ThreeD: TypeAlias = THREE

Point: TypeAlias = TypedNDArray[tuple[DimSpace]]
Vector: TypeAlias = TypedNDArray[tuple[DimSpace]]
RotationMatrix: TypeAlias = TypedNDArray[tuple[DimSpace, DimSpace]]

ScalarArray: TypeAlias = TypedNDArray[tuple[NumPoints]]
PointSet: TypeAlias = TypedNDArray[tuple[NumPoints, DimSpace]]

Jacobian: TypeAlias = TypedNDArray[tuple[DimSpace, DimSpace]]
JacobianSet: TypeAlias = TypedNDArray[tuple[NumPoints, DimSpace, DimSpace]]


class LearnableEndomorphicMappingProtocol(Protocol[DimSpace]):
    """A generic learnable mapping interface denoting `R^DimSpace -> R^DimSpace`."""

    def fit(
        self,
        source_points: PointSet[NumPoints, DimSpace],
        target_points: PointSet[NumPoints, DimSpace],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def predict(
        self, points: PointSet[NumPoints, DimSpace], /, *args: Any, **kwargs: Any
    ) -> PointSet[NumPoints, DimSpace]: ...

    def jacobian(
        self, points: PointSet[NumPoints, DimSpace], /, *args: Any, **kwargs: Any
    ) -> JacobianSet[NumPoints, DimSpace]: ...
