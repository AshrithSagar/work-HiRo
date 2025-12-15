"""
Base Transform
=======
src/tp_gpt/transforms/base.py
"""

from typing import Any, Protocol

from typed_numpy.helpers import ArrayNx2, ArrayNx2x2, ArrayNx3, ArrayNx3x3

from tp_gpt.typings import JacobianT, PointsT


class Transform(Protocol[PointsT, JacobianT]):
    """A generic transform interface"""

    PointsClass: type[PointsT]
    JacobianClass: type[JacobianT]

    def fit(
        self,
        source_points: PointsT,
        target_points: PointsT,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def predict(self, points: PointsT, /, *args: Any, **kwargs: Any) -> PointsT: ...

    def jacobian(self, points: PointsT, /, *args: Any, **kwargs: Any) -> JacobianT: ...


class Transform2D(Transform[ArrayNx2, ArrayNx2x2], Protocol):
    """A generic 2D transform interface"""


class Transform3D(Transform[ArrayNx3, ArrayNx3x3], Protocol):
    """A generic 3D transform interface"""
