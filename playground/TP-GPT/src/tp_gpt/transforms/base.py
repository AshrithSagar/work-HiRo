"""
Base Transform
=======
src/tp_gpt/transforms/base.py
"""

from typing import Any, Protocol

from tp_gpt.typings import THREE, TWO, DimDT, DimNT, JacobianArray, PointsArray


class Transform(Protocol[DimNT, DimDT]):
    """A generic transform interface"""

    def fit(
        self,
        source_points: PointsArray[DimNT, DimDT],
        target_points: PointsArray[DimNT, DimDT],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def predict(
        self, points: PointsArray[DimNT, DimDT], /, *args: Any, **kwargs: Any
    ) -> PointsArray[DimNT, DimDT]: ...

    def jacobian(
        self, points: PointsArray[DimNT, DimDT], /, *args: Any, **kwargs: Any
    ) -> JacobianArray[DimNT, DimDT]: ...


class Transform2D(Transform[DimNT, TWO], Protocol):
    """A generic 2D transform interface"""


class Transform3D(Transform[DimNT, THREE], Protocol):
    """A generic 3D transform interface"""
