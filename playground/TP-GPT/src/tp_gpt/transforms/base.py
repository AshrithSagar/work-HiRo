"""
Base Transform
=======
src/tp_gpt/transforms/base.py
"""

from typing import Any, Protocol

from tp_gpt.core.typings import (
    DimSpace,
    JacobianArray,
    NumPoints,
    PointSet,
    ThreeD,
    TwoD,
)


class Transform(Protocol[NumPoints, DimSpace]):
    """A generic transform interface"""

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
    ) -> JacobianArray[NumPoints, DimSpace]: ...


class Transform2D(Transform[NumPoints, TwoD], Protocol):
    """A generic 2D transform interface"""


class Transform3D(Transform[NumPoints, ThreeD], Protocol):
    """A generic 3D transform interface"""
