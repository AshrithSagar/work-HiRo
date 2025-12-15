"""
Typing utils
=======
src/tp_gpt/typings.py
"""

from typing import TypeVar

from typed_numpy.helpers import Array1D, Array2D, Array3D

PointT = TypeVar("PointT", bound=Array1D, default=Array1D)
PointsT = TypeVar("PointsT", bound=Array2D, default=Array2D)
RotationT = TypeVar("RotationT", bound=Array2D, default=Array2D)
JacobianT = TypeVar("JacobianT", bound=Array3D, default=Array3D)
