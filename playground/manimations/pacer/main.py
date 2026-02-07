from typing import Literal, TypeAlias, TypeVar

import manim as mn  # type: ignore
import numpy as np
import pyLasaDataset as lasa  # type: ignore
from pyLasaDataset.dataset import _Data  # type: ignore
from typed_numpy._typed import TypedNDArray
from typed_numpy._typed.shapes import THREE, TWO

THOUSAND = Literal[1000]
DType: TypeAlias = np.float64
Dim1 = TypeVar("Dim1", bound=int, default=int)
Dim2 = TypeVar("Dim2", bound=int, default=int)
Array1D: TypeAlias = TypedNDArray[tuple[Dim1], np.dtype[DType]]
Array2D: TypeAlias = TypedNDArray[tuple[Dim1, Dim2], np.dtype[DType]]

NumPoints = TypeVar("NumPoints", bound=int, default=int)
Demo: TypeAlias = Array2D[TWO, THOUSAND]
Point2D: TypeAlias = Array1D[TWO]
Point3D: TypeAlias = Array1D[THREE]
Points2D: TypeAlias = Array2D[THOUSAND, TWO]
Points3D: TypeAlias = Array2D[THOUSAND, THREE]


class DemonstrationScene(mn.Scene):
    def setup(self) -> None:
        assert isinstance(self.camera, mn.Camera)
        self.camera.background_color = mn.BLACK

    def construct(self) -> None:
        self.draw_demo()

    def draw_demo(
        self,
        data: _Data = lasa.DataSet.GShape,
        demo_idx: int = 0,  # i
    ) -> None:
        demo = Demo(data.demos[demo_idx].__getattribute__("pos"))
        n_points = demo.shape[1]  # T_i
        points = Points3D([(demo[0, t], demo[1, t], 0) for t in range(n_points)])

        curve = mn.VMobject()
        curve.set_points_smoothly(points)
        curve.center()
        curve.scale_to_fit_width(5)  # type: ignore
        self.add(curve)
        self.play(mn.Create(curve), run_time=4)
        self.wait()
