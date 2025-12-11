"""
Policy transportation
=========
src/tp_gpt/transportation.py
"""

from typing import Any, Protocol

from typed_numpy import TypedNDArray


class NonLinearTransform(Protocol):
    def fit(self, *args, **kwargs) -> Any: ...
    def predict(self, *args, **kwargs) -> Any: ...


class PolicyTransportation[NLT: NonLinearTransform]:
    def __init__(self, nonlinear_transform: NLT):
        self.nonlinear_transform = nonlinear_transform

    def fit(self, source_points: TypedNDArray, target_points: TypedNDArray):
        self.nonlinear_transform.fit(source_points, target_points)

    def predict(self, points: TypedNDArray) -> TypedNDArray:
        return self.nonlinear_transform.predict(points)
