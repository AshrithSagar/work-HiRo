"""
Mapping abstractions
=======
src/tp_gpt/core/mappings.py
"""

from typing import Any, Protocol, TypeAlias

from typed_numpy._typed import ShapedNDArray as Shaped
from typed_numpy._typed import TypedNDArray as NDArray

from tp_gpt.core.spaces import CoordinateSpace, PointSet, SpaceCollection
from tp_gpt.core.typings import DimSpace, NumPoints, _DimSpace, _NumPoints

Jacobian: TypeAlias = NDArray[tuple[DimSpace, DimSpace]]
JacobianSet: TypeAlias = NDArray[tuple[NumPoints, DimSpace, DimSpace]]


class EndomorphicMapping(CoordinateSpace[DimSpace]):
    _DimSpace = _DimSpace

    _Jacobian: Shaped[tuple[DimSpace, DimSpace]] = Shaped(_DimSpace, _DimSpace)


class EndomorphicMappingCollection(
    SpaceCollection[NumPoints, DimSpace], EndomorphicMapping[DimSpace]
):
    _NumPoints = _NumPoints
    _DimSpace = _DimSpace

    _JacobianSet: Shaped[tuple[NumPoints, DimSpace, DimSpace]] = Shaped(
        _NumPoints, _DimSpace, _DimSpace
    )


class LearnableEndomorphicMappingProtocol(Protocol[NumPoints, DimSpace]):
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
