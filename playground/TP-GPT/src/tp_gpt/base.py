"""
Base module
=======
src/tp_gpt/base.py
"""

from typing import Any, Protocol

from numpy.typing import NDArray


class Transform(Protocol):
    """A generic transform interface"""

    def fit(
        self,
        source_points: NDArray,
        target_points: NDArray,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def predict(self, points: NDArray, /, *args: Any, **kwargs: Any) -> NDArray: ...

    def jacobian(self, points: NDArray, /, *args: Any, **kwargs: Any) -> NDArray: ...
