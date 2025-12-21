"""
Typing utils
=======
src/tp_gpt/core/typings.py
"""

from typing import TypeAlias, TypeVar

from typed_numpy._typed import DimVar
from typed_numpy._typed.shapes import THREE, TWO

## Static bindings

TwoD: TypeAlias = TWO
ThreeD: TypeAlias = THREE

DimSpace = TypeVar("DimSpace", bound=int, default=int)
"""TypeVar denoting dimension of the space"""

NumPoints = TypeVar("NumPoints", bound=int, default=int)
"""TypeVar denoting number of points"""


## Runtime bindings

_DimSpace = DimVar()
"""DimVar denoting dimension of the space"""

_NumPoints = DimVar()
"""DimVar denoting number of points"""
