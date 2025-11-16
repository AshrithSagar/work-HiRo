"""
Types
=======
src/tp_gpt/typings.py
"""

from typing import Any, TypeAlias

import numpy as np

type NDArray[
    ST: tuple[int, ...] = tuple[Any, ...],
    DT: np.dtype = np.dtype[np.generic],
] = np.ndarray[ST, DT]
"""
Generic type alias type for `numpy.ndarray`.
Stronger typing than `numpy.typing.NDArray`.
"""

# Shape type aliases
Shape1D: TypeAlias = tuple[int]
"""A tuple representing shape `(N,)`."""
Shape2D: TypeAlias = tuple[int, int]
"""A tuple representing shape `(M, N)`."""
Shape3D: TypeAlias = tuple[int, int, int]
"""A tuple representing shape `(L, M, N)`."""
Shape4D: TypeAlias = tuple[int, int, int, int]
"""A tuple representing shape `(K, L, M, N)`."""
ShapeND: TypeAlias = tuple[int, ...]
"""A tuple representing shape `(N, ...)`."""

# Array type aliases
Array1D: TypeAlias = NDArray[Shape1D, np.dtype[np.double]]
"""A `numpy.ndarray` of shape `(N,)` with dtype `np.double`."""
Array2D: TypeAlias = NDArray[Shape2D, np.dtype[np.double]]
"""A `numpy.ndarray` of shape `(M, N)` with dtype `np.double`."""
Array3D: TypeAlias = NDArray[Shape3D, np.dtype[np.double]]
"""A `numpy.ndarray` of shape `(L, M, N)` with dtype `np.double`."""
Array4D: TypeAlias = NDArray[Shape4D, np.dtype[np.double]]
"""A `numpy.ndarray` of shape `(K, L, M, N)` with dtype `np.double`."""
ArrayND: TypeAlias = NDArray[ShapeND, np.dtype[np.double]]
"""A `numpy.ndarray` of shape `(N, ...)` with dtype `np.double`."""
