from typing import Literal, TypeAlias, TypeVar

import manim as mn  # type: ignore
import manim.typing as mnt  # type: ignore
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


class DemonstrationScene(mn.Scene):
    def setup(self) -> None:
        assert isinstance(self.camera, mn.Camera)
        self.camera.background_color = mn.BLACK

    def construct(self) -> None:
        # ── Demonstrations ───────────────────────────────────────────────────────
        # Draw demonstrations into scene

        data = lasa.DataSet.GShape
        n_demos = len(data.demos)  # N
        demo_indices = list(range(n_demos))  # All demonstrations

        curves = list[mn.VMobject]()
        curve_colors = mn.color_gradient(
            [mn.RED, mn.ORANGE, mn.YELLOW, mn.GREEN, mn.BLUE, mn.PURPLE], n_demos
        )
        for i in demo_indices:
            demo = Demo(data.demos[i].__getattribute__("pos"))
            n_points = demo.shape[1]  # T_i
            points = Points3D([(demo[0, t], demo[1, t], 0.0) for t in range(n_points)])

            curve = mn.VMobject()
            curve.set_points_smoothly(points)
            curve.set_stroke(color=curve_colors[i], width=3, opacity=0.7)
            curves.append(curve)

        group = mn.VGroup(*curves)
        group.center()
        group.scale_to_fit_width(5)  # type: ignore
        self.play(
            *[mn.Create(curve) for curve in curves],
            run_time=3,
        )
        self.wait()

        # ── Phase slider ─────────────────────────────────────────────────────────
        # Animate phase variable on the demonstrations

        tau = mn.ValueTracker(0)

        curve_dots = list[mn.Mobject]()
        for curve in curves:
            dot = mn.always_redraw(
                lambda curve=curve: mn.Dot(
                    curve.point_from_proportion(tau.get_value()),
                    fill_opacity=curve.stroke_opacity,
                    color=curve.color,
                )
            )
            curve_dots.append(dot)
        self.add(*curve_dots)

        progress_line = mn.NumberLine(
            x_range=[0, 1, 0.1],
            length=10,
            include_numbers=True,
        ).to_edge(mn.DOWN)
        progress_dot = mn.always_redraw(
            lambda: mn.Dot(progress_line.n2p(tau.get_value()), color=mn.YELLOW)
        )
        tau_label = mn.MathTex(r"\tau", color=mn.WHITE)
        tau_label.next_to(progress_line, mn.LEFT)
        self.add(progress_line, progress_dot, tau_label)

        self.wait()
        self.play(tau.animate.set_value(1), run_time=4, rate_func=mn.linear)
        self.wait()

        # ── Binning ──────────────────────────────────────────────────────────────
        # Divide phase into B equal bins

        for dot in curve_dots:
            dot.clear_updaters()
        self.remove(*curve_dots)
        progress_dot.clear_updaters()
        self.remove(progress_dot)
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

        self.play(
            *[mn.FadeIn(line) for line in bin_lines],
            run_time=1,
        )
        self.wait()

        ## Curve bins
        segmented_curves_per_curve = list[mn.VGroup]()
        for curve in curves:
            segments = list[mn.VMobject]()
            for b in range(n_bins):
                t0, t1 = b / n_bins, (b + 1) / n_bins
                segment: mn.VMobject = mn.VMobject()
                segment.set_points_smoothly(
                    [
                        curve.point_from_proportion(alpha)
                        for alpha in np.linspace(t0, t1, 30)
                    ]
                )
                segment.set_stroke(color=mn.WHITE, width=3, opacity=0.7)
                segments.append(segment)
            segmented_curves_per_curve.append(mn.VGroup(*segments))

        for seg_group in segmented_curves_per_curve:
            self.add(*seg_group)

        # Create a single blob per bin
        bin_blobs = list[mn.Rectangle]()
        for line, color in zip(bin_lines, bin_colors):
            blob: mn.Rectangle = mn.Rectangle(
                width=line.get_length(),
                height=0.3,
                fill_color=color,
                fill_opacity=0.6,
                stroke_opacity=0,
            ).move_to(line.get_top() + mn.UP * 0.15)
            bin_blobs.append(blob)
            self.add(blob)

        # Animate each blob to "flood-fill" all segments of its bin
        for b, blob in enumerate(bin_blobs):
            # Target position: average center of all segments in this bin
            segment_centers: list[mnt.Point3D] = [
                seg_group[b].get_center()  # type: ignore
                for seg_group in segmented_curves_per_curve
            ]
            avg_center = Point3D(sum(segment_centers) / len(segment_centers))

            self.play(blob.animate.move_to(avg_center), run_time=0.8)
            for seg_group in segmented_curves_per_curve:
                seg_group[b].set_color(bin_colors[b])  # type: ignore
            self.wait(0.3)

        self.play(*[mn.FadeOut(blob) for blob in bin_blobs], run_time=0.5)
        self.wait()

        # ─────────────────────────────────────────────────────────────────────────
