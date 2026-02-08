from typing import Literal, TypeAlias, TypeVar

import manim as mn  # type: ignore
import numpy as np
import pyLasaDataset as lasa  # type: ignore
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


def dot_on_curve(
    curve: mn.VMobject,
    tau: mn.ValueTracker,
    *,
    color: mn.ManimColor,
    opacity: float,
    radius: float = mn.DEFAULT_DOT_RADIUS,
) -> mn.Dot:
    dot = mn.Dot(
        curve.point_from_proportion(0),
        color=color,
        fill_opacity=opacity,
        radius=radius,
    )

    def updater(m: mn.Dot) -> None:
        m.move_to(curve.point_from_proportion(tau.get_value()))

    dot.add_updater(updater)  # type: ignore
    return dot


def velocity_arrow_on_curve(
    curve: mn.VMobject,
    pos: Demo,
    vel: Demo,
    tau: mn.ValueTracker,
    *,
    color: mn.ManimColor,
    vec_scale: float,
    eps: float = 1e-6,
) -> mn.Arrow:
    start = curve.get_start()
    arrow = mn.Arrow(
        start=start,
        end=start + mn.RIGHT * eps,
        buff=0,
        stroke_width=3,
        color=color,
    )

    T = pos.shape[1]

    def updater(m: mn.Arrow) -> None:
        idx = min(int(tau.get_value() * (T - 1)), T - 1)

        p = curve.point_from_proportion(idx / (T - 1))
        v = vec_scale * np.array([vel[0, idx], vel[1, idx], 0.0])

        norm = np.linalg.norm(v)
        if norm < eps:
            m.set_opacity(0.0)
            m.put_start_and_end_on(p, p + mn.RIGHT * eps)
            return

        m.set_opacity(1.0)
        m.put_start_and_end_on(p, p + v)

    arrow.add_updater(updater)  # type: ignore
    return arrow


def slider_dot(
    line: mn.NumberLine,
    tau: mn.ValueTracker,
    *,
    color: mn.ManimColor = mn.YELLOW,
) -> mn.Dot:
    dot = mn.Dot(line.n2p(0), color=color)

    def updater(m: mn.Dot) -> None:
        m.move_to(line.n2p(tau.get_value()))

    dot.add_updater(updater)  # type: ignore
    return dot


def clear_and_remove(scene: mn.Scene, *mobjects: mn.Mobject) -> None:
    for m in mobjects:
        m.clear_updaters()
    scene.remove(*mobjects)


class DemonstrationScene(mn.Scene):
    def setup(self) -> None:
        assert isinstance(self.camera, mn.Camera)
        self.camera.background_color = mn.BLACK

    def construct(self) -> None:
        # ── Demonstrations ───────────────────────────────────────────────────────
        # Draw demonstrations into scene
        self.next_section(skip_animations=False)

        data = lasa.DataSet.GShape
        n_demos = len(data.demos)  # N
        demo_indices = list(range(n_demos))  # All demonstrations

        curves = list[mn.VMobject]()
        curve_colors = mn.color_gradient(
            [mn.RED, mn.ORANGE, mn.YELLOW, mn.GREEN, mn.BLUE, mn.PURPLE], n_demos
        )
        for i in demo_indices:
            pos = Demo(data.demos[i].__getattribute__("pos"))
            n_points = pos.shape[1]  # T_i
            points = Points3D([(pos[0, t], pos[1, t], 0.0) for t in range(n_points)])

            curve = mn.VMobject()
            curve.set_points_smoothly(points)
            curve.set_stroke(color=curve_colors[i], width=3, opacity=0.7)
            curves.append(curve)

        curves_group = mn.VGroup(*curves)
        curves_group.center()
        curves_group.scale_to_fit_width(5)  # type: ignore
        self.play(
            *[mn.Create(curve) for curve in curves],
            run_time=3,
        )
        self.wait()

        # ── Phase slider ─────────────────────────────────────────────────────────
        # Animate phase variable on the demonstrations
        self.next_section(skip_animations=False)

        tau = mn.ValueTracker(0)

        curve_dots = list[mn.Dot]()
        for curve in curves:
            dot = dot_on_curve(
                curve,
                tau,
                color=curve.color,
                opacity=curve.stroke_opacity,
            )
            curve_dots.append(dot)
        self.add(*curve_dots)

        velocity_vectors = list[mn.Arrow]()
        vec_scale = 0.03  # Purely for visual
        for i, curve in enumerate(curves):
            pos = Demo(data.demos[i].__getattribute__("pos"))
            vel = Demo(data.demos[i].__getattribute__("vel"))
            vector = velocity_arrow_on_curve(
                curve,
                pos,
                vel,
                tau,
                color=curve.color,
                vec_scale=vec_scale,
            )
            velocity_vectors.append(vector)
        self.add(*velocity_vectors)

        progress_line = mn.NumberLine(
            x_range=[0, 1, 0.1],
            length=10,
            include_numbers=True,
        ).to_edge(mn.DOWN)
        progress_dot = slider_dot(progress_line, tau)
        tau_label = mn.MathTex(r"\tau", color=mn.WHITE)
        tau_label.next_to(progress_line, mn.LEFT)
        self.add(progress_line, progress_dot, tau_label)

        self.wait()
        self.play(tau.animate.set_value(1), run_time=4, rate_func=mn.linear)
        self.wait()

        # ── Binning ──────────────────────────────────────────────────────────────
        # Divide phase into B equal bins
        self.next_section(skip_animations=False)

        clear_and_remove(self, *curve_dots)
        clear_and_remove(self, progress_dot)
        for curve in curves:
            curve.set_color(mn.WHITE)

        n_bins = 5  # B
        bin_colors = mn.color_gradient(
            [mn.RED, mn.ORANGE, mn.YELLOW, mn.GREEN, mn.BLUE], n_bins
        )

        ## Phase slider bins (on the number line)
        bin_lines = list[mn.Mobject]()
        for b in range(n_bins):
            t0, t1 = b / n_bins, (b + 1) / n_bins
            line = mn.Line(
                progress_line.n2p(t0),
                progress_line.n2p(t1),
                stroke_width=36,
                color=bin_colors[b],
                stroke_opacity=0.5,
            )
            bin_lines.append(line)

        ## Curve bins
        # Segment the curves and prepare them for fade-in
        segmented_curves_per_curve = list[mn.VGroup]()
        all_segments = list[mn.VMobject]()
        for curve in curves:
            segments = list[mn.VMobject]()
            for b in range(n_bins):
                t0, t1 = b / n_bins, (b + 1) / n_bins
                segment = mn.VMobject()
                segment.set_points_smoothly(
                    [
                        curve.point_from_proportion(alpha)
                        for alpha in np.linspace(t0, t1, 10)
                    ]
                )
                segment.set_stroke(color=bin_colors[b], width=3, opacity=1.0)
                segments.append(segment)
                all_segments.append(segment)
            segmented_curves_per_curve.append(mn.VGroup(*segments))

        self.add(*bin_lines)
        self.add(*all_segments)
        self.play(
            *[mn.FadeIn(line, run_time=1) for line in bin_lines]
            + [mn.FadeIn(seg, run_time=1) for seg in all_segments]
        )
        self.wait()

        # ─────────────────────────────────────────────────────────────────────────
