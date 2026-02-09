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
        # ── Introduction ─────────────────────────────────────────────────────────
        # Title introduction
        self.next_section(skip_animations=False)

        heading = mn.Text("PACER", font_size=48, color=mn.WHITE)
        caption = mn.Text(
            "Progress-Aligned Curation for Error-Resilient Imitation Learning",
            font_size=32,
            color=mn.LIGHT_GRAY,
            line_spacing=1.2,
        )
        caption.next_to(heading, mn.DOWN, buff=0.5)

        self.play(mn.FadeIn(heading, shift=mn.DOWN * 0.5))
        self.play(mn.Write(caption), run_time=2)
        self.wait()

        # ── Demonstrations ───────────────────────────────────────────────────────
        # Draw demonstrations into scene
        self.next_section(skip_animations=False)

        old_heading = heading
        heading = mn.Text("Demonstrations", font_size=36, color=mn.WHITE)
        heading.to_corner(mn.UL)
        self.play(mn.ReplacementTransform(mn.VGroup(old_heading, caption), heading))

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

        # ── Phase slider ─────────────────────────────────────────────────────────
        # Animate phase variable on the demonstrations
        self.next_section(skip_animations=False)

        old_heading = heading
        heading = mn.Text("Phase Alignment", font_size=36, color=mn.WHITE)
        heading.to_corner(mn.UL)
        self.play(mn.ReplacementTransform(old_heading, heading))

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

        velocity_vectors = list[mn.Mobject]()
        vec_scale = 0.03  # Purely for visual
        for i, curve in enumerate(curves):
            pos = Demo(data.demos[i].__getattribute__("pos"))
            vel = Demo(data.demos[i].__getattribute__("vel"))
            vector = mn.always_redraw(
                lambda pos=pos, vel=vel, curve=curve: (
                    lambda idx: (
                        lambda p, v: mn.Arrow(
                            start=p,
                            end=p + v,
                            buff=0,
                            stroke_width=3,
                            color=curve.color,
                        )
                    )(
                        curve.point_from_proportion(idx / (pos.shape[1] - 1)),
                        vec_scale * np.array([vel[0, idx], vel[1, idx], 0.0]),
                    )
                )(min(int(tau.get_value() * (pos.shape[1] - 1)), pos.shape[1] - 1))
            )
            velocity_vectors.append(vector)
        self.add(*velocity_vectors)

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
        self.next_section(skip_animations=False)

        for dot in curve_dots:
            dot.clear_updaters()
        self.remove(*curve_dots)
        progress_dot.clear_updaters()
        self.remove(progress_dot)
        for curve in curves:
            curve.set_color(mn.WHITE)

        old_heading = heading
        heading = mn.Text("Binning", font_size=36, color=mn.WHITE)
        heading.to_corner(mn.UL)
        self.play(mn.ReplacementTransform(old_heading, heading))

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
        segmented_group = mn.VGroup(*all_segments)

        self.play(
            *[mn.FadeIn(line) for line in bin_lines]
            + [mn.FadeIn(seg) for seg in all_segments]
            + [mn.FadeOut(curve) for curve in curves]
        )

        center = curves_group.get_center()

        def radial_shift(mobject: mn.Mobject, strength: float = 0.4) -> mnt.Vector3D:
            direction = mobject.get_center() - center
            if np.linalg.norm(direction) == 0:
                return mn.ORIGIN
            return strength * direction / np.linalg.norm(direction)

        segments_per_bin = [
            mn.VGroup(*(segmented_curves_per_curve[i][b] for i in range(len(curves))))  # type: ignore
            for b in range(n_bins)
        ]
        target_group = segmented_group.copy()
        for b, bg in enumerate(segments_per_bin):
            target_bin = mn.VGroup(*target_group.submobjects[b::n_bins])
            target_bin.shift(radial_shift(bg))
        target_group.scale_to_fit_width(5)  # type: ignore
        self.play(mn.Transform(segmented_group, target_group))
        self.wait()

        # ── Robust Consensus Statistics ──────────────────────────────────────────
        # Animate consensus computation for a particular bin
        self.next_section(skip_animations=False)

        old_heading = heading
        heading = mn.Text("Robust Consensus Statistics", font_size=36, color=mn.WHITE)
        heading.to_corner(mn.UL)
        self.play(mn.ReplacementTransform(old_heading, heading))

        bin_idx = 0
        bin_group = segments_per_bin[bin_idx]
        self.play(
            *[
                seg.animate.set_stroke(opacity=0.25)
                for seg in segmented_group
                if seg not in bin_group.submobjects
            ],
        )

        shift_vec = mn.ORIGIN - bin_group.get_center()
        self.play(
            segmented_group.animate.shift(shift_vec).scale(3, about_point=mn.ORIGIN)
        )

        self.wait()

        # ─────────────────────────────────────────────────────────────────────────
