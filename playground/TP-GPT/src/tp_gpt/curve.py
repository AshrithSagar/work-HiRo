"""
Curve utils
=========
src/tp_gpt/curve.py
"""

import numpy as np
import numpy.typing as npt
from typed_numpy.helpers import Array2, ArrayN, ArrayNx2


class Curve:
    def __init__(self, xs: npt.ArrayLike, ys: npt.ArrayLike):
        self.xs = ArrayN(xs)
        self.ys = ArrayN(ys)
        self.points = ArrayNx2(np.column_stack((self.xs, self.ys)))
        self.n_points = len(self.points)

    def __getitem__(self, idx: int) -> Array2:
        return self.points[idx]

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Curve(xs={self.xs}, ys={self.ys}, n_points={self.n_points})"

    @property
    def start_pt(self) -> Array2:
        return Array2([self.xs[0], self.ys[0]])

    @property
    def end_pt(self) -> Array2:
        return Array2([self.xs[-1], self.ys[-1]])

    @classmethod
    def from_points(cls, points: npt.ArrayLike) -> "Curve":
        points_arr = ArrayNx2(points)
        return cls(xs=points_arr[:, 0], ys=points_arr[:, 1])
