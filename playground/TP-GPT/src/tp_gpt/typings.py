"""
Types
=======
src/tp_gpt/typings.py
"""

from typing import Annotated, Any, TypeAlias

import numpy as np
import numpy.typing as npt
from beartype import beartype
from jaxtyping import Float64, jaxtyped

## Defaults
dtype: TypeAlias = np.double
"""The default `dtype` used throughout, mostly."""

ArrayDType = Float64
"""The default `jaxtyping` dtype used throughout, mostly."""

## ===== Static =====

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
Array1D: TypeAlias = NDArray[Shape1D, np.dtype[dtype]]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""
Array2D: TypeAlias = NDArray[Shape2D, np.dtype[dtype]]
"""A `numpy.ndarray` of shape `(M, N)` with the default `dtype`."""
Array3D: TypeAlias = NDArray[Shape3D, np.dtype[dtype]]
"""A `numpy.ndarray` of shape `(L, M, N)` with the default `dtype`."""
Array4D: TypeAlias = NDArray[Shape4D, np.dtype[dtype]]
"""A `numpy.ndarray` of shape `(K, L, M, N)` with the default `dtype`."""
ArrayND: TypeAlias = NDArray[ShapeND, np.dtype[dtype]]
"""A `numpy.ndarray` of shape `(N, ...)` with the default `dtype`."""


# Helpers
def asArray(arr: npt.ArrayLike, dtype: npt.DTypeLike = dtype):
    """Wrapper for `numpy.asarray`. Uses the default `dtype` if not specified."""
    return np.asarray(arr, dtype=dtype)


## ===== Runtime =====

runtime_typecheck = jaxtyped(typechecker=beartype)
"""A decorator for runtime type checking using jaxtyping and beartype."""

type Array[shape: str] = ArrayDType[np.ndarray, shape]
"""A generic wrapper for `jaxtyping` for `numpy.ndarray`."""

# Array type aliases
Array2x2 = Annotated[Array2D, Array["2 2"]]
ArrayN = Annotated[Array1D, Array["N"]]
ArrayNx2 = Annotated[Array2D, Array["N 2"]]
