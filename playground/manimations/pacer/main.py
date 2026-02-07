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
        curves = self.draw_demos(lasa.DataSet.GShape)
        self.animate_phase_slider(curves)

    def draw_demos(
        self, data: _Data, demo_indices: list[int] | None = None
    ) -> list[mn.VMobject]:
        n_demos = len(data.demos)  # N
        if demo_indices is None:
            demo_indices = list(range(n_demos))

        curves = list[mn.VMobject]()
        colors = mn.color_gradient(
            [mn.RED, mn.ORANGE, mn.YELLOW, mn.GREEN, mn.BLUE, mn.PURPLE], n_demos
        )
        for i in demo_indices:
            demo = Demo(data.demos[i].__getattribute__("pos"))
            n_points = demo.shape[1]  # T_i
            points = Points3D([(demo[0, t], demo[1, t], 0.0) for t in range(n_points)])

            curve = mn.VMobject()
            curve.set_points_smoothly(points)
            curve.set_stroke(color=colors[i], width=3, opacity=0.7)
            curves.append(curve)

        group = mn.VGroup(*curves)
        group.center()
        group.scale_to_fit_width(5)  # type: ignore
        self.play(
            *[mn.Create(curve) for curve in curves],
            run_time=3,
        )
        self.wait()
        return curves

    def animate_phase_slider(self, curves: list[mn.VMobject]) -> None:
        tau = mn.ValueTracker(0)

        dots = list[mn.Mobject]()
        for curve in curves:
            dot = mn.always_redraw(
                lambda curve=curve: mn.Dot(
                    curve.point_from_proportion(tau.get_value()),
                    fill_opacity=curve.stroke_opacity,
                    color=curve.color,
                )
            )
            dots.append(dot)
        self.add(*dots)

        progress_line = mn.NumberLine(
            x_range=[0, 1, 0.1],
            length=10,
            include_numbers=True,
        ).to_edge(mn.DOWN)
        progress_dot = mn.always_redraw(
            lambda: mn.Dot(progress_line.n2p(tau.get_value()), color=mn.YELLOW)
        )
        self.add(progress_line, progress_dot)

        self.wait()
        self.play(tau.animate.set_value(1), run_time=4, rate_func=mn.linear)
        self.wait()
