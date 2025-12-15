"""
Base Transform
=======
src/tp_gpt/transforms/base.py
"""

from typing import Any, Protocol

from tp_gpt.typings import DimT, JacobianArray, PointsArray, ThreeD, TwoD


class Transform(Protocol[DimT]):
    """A generic transform interface"""

    def fit(
        self,
        source_points: PointsArray[DimT],
        target_points: PointsArray[DimT],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def predict(
        self, points: PointsArray[DimT], /, *args: Any, **kwargs: Any
    ) -> PointsArray[DimT]: ...

    def jacobian(
        self, points: PointsArray[DimT], /, *args: Any, **kwargs: Any
    ) -> JacobianArray[DimT]: ...


class Transform2D(Transform[TwoD], Protocol):
    """A generic 2D transform interface"""


class Transform3D(Transform[ThreeD], Protocol):
    """A generic 3D transform interface"""
