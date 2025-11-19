"""
Types
=======
src/tp_gpt/typings.py
"""

from typing import Any, Literal, TypeAlias, TypeVar, get_args, get_origin

import numpy as np
import numpy.typing as npt

## Typed NDArray

# `numpy` privates
_Shape: TypeAlias = tuple[Any, ...]  # Weakened type reduction
_AnyShape: TypeAlias = tuple[Any, ...]

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)


class DimensionError(Exception): ...


class ShapeError(Exception): ...


_AcceptedDim: TypeAlias = int | TypeVar | Any | None
_RuntimeDim: TypeAlias = int | None
_RuntimeShape: TypeAlias = tuple[_RuntimeDim, ...]


def _normalise_dim(dim: _AcceptedDim) -> _RuntimeDim:
    """Normalise a dimension specifier into something that can be runtime-validated."""

    if dim is None:
        return None
    if isinstance(dim, int):
        return dim
    if isinstance(dim, TypeVar):
        return None

    origin = get_origin(dim)
    if origin is Literal:
        lit = get_args(dim)
        if len(lit) == 1 and isinstance(lit[0], int):
            return lit[0]

    if dim is Any:
        return None
    return None  # Fallback


def _normalise_shape(shape: _Shape) -> _RuntimeShape:
    return tuple(_normalise_dim(dim) for dim in shape)


class NDArray(np.ndarray[_ShapeT_co, _DTypeT_co]):
    """Generic `numpy.ndarray` subclass with static shape typing and runtime shape validation."""

    __shape__: _RuntimeShape | None = None
    """Runtime shape metadata."""

    def __new__(
        cls,
        arr: npt.ArrayLike,
        *,
        dtype: npt.DTypeLike | None = None,
        shape: _ShapeT_co | None = None,
    ) -> "NDArray[_ShapeT_co, _DTypeT_co]":
        _arr: np.ndarray[tuple[int, ...]] = np.asarray(arr, dtype=dtype)
        obj = _arr.view(cls)

        # Set metadata
        obj.__shape__ = _normalise_shape(shape) if shape is not None else None

        # Runtime validation
        if obj.__shape__ is not None:
            expected = obj.__shape__
            actual = _arr.shape

            if len(expected) != len(actual):
                raise DimensionError(
                    f"Dimension mismatch: expected {len(expected)}, got {len(actual)}"
                )

            for exp, act in zip(expected, actual):
                if exp is not None and exp != act:
                    raise ShapeError(
                        f"Shape mismatch: expected {expected}, got {actual}"
                    )

        return obj

    def __array_finalize__(self, obj: npt.NDArray[Any] | None, /) -> None:
        if obj is None:
            return

        # Propagate metadata
        self.__shape__ = getattr(obj, "__shape__", None)


## Helpers

def_dtype: TypeAlias = np.double
"""The default `dtype` used throughout, mostly."""

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
Array1D: TypeAlias = NDArray[Shape1D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""
Array2D: TypeAlias = NDArray[Shape2D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(M, N)` with the default `dtype`."""
Array3D: TypeAlias = NDArray[Shape3D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(L, M, N)` with the default `dtype`."""
Array4D: TypeAlias = NDArray[Shape4D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(K, L, M, N)` with the default `dtype`."""
ArrayND: TypeAlias = NDArray[ShapeND, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, ...)` with the default `dtype`."""

TWO: TypeAlias = Literal[2]
"""Literal type for the integer `2`."""

Array2x2: TypeAlias = NDArray[tuple[TWO, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 2)` with the default `dtype`."""
ArrayN: TypeAlias = NDArray[tuple[int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""
ArrayNx2: TypeAlias = NDArray[tuple[int, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 2)` with the default `dtype`."""
