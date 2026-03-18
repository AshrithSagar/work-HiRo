"""
PACER manimation
=======
"""
# src/manimations/main.py

import manim as mn
import manim.typing as mnt
import numpy as np

from pacer.datasets import LASADataSet3D


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

        data = LASADataSet3D("GShape", dtype=np.float64)

        curves = list[mn.VMobject]()
        curve_colors = mn.color_gradient(
            [mn.RED, mn.ORANGE, mn.YELLOW, mn.GREEN, mn.BLUE, mn.PURPLE], len(data)
        )
        for points, color in zip(data.positions, curve_colors):
            curve = mn.VMobject()
            curve.set_points_smoothly(points)
            curve.set_stroke(color=color, width=3, opacity=0.7)
            curves.append(curve)

        curves_group = mn.VGroup(*curves)
        curves_group.center()
        curves_group.scale_to_fit_width(5)  # pyright: ignore[reportUnknownMemberType]
        self.play(*[mn.Create(curve) for curve in curves])

        # ── Phase slider ─────────────────────────────────────────────────────────
        # Animate phase variable on the demonstrations
        self.next_section(skip_animations=False)

        old_heading = heading
        heading = mn.Text("Phase", font_size=36, color=mn.WHITE)
        heading.to_corner(mn.UL)
        self.play(mn.ReplacementTransform(old_heading, heading))

        tau = mn.ValueTracker(0)

        curve_dots = list[mn.Dot]()
        velocity_vectors = list[mn.Arrow]()
        vec_scale = 0.03  # Purely for visual
        for curve in curves:
            dot = mn.Dot(
                point=curve.point_from_proportion(0),
                fill_opacity=curve.stroke_opacity,
                color=curve.color,
            )
            curve_dots.append(dot)
            arrow = mn.Arrow(
                start=curve.point_from_proportion(0),
                end=curve.point_from_proportion(0) + 1e-3 * mn.RIGHT,
                buff=0,
                stroke_width=3,
                color=curve.color,
            )
            velocity_vectors.append(arrow)
        curve_dots_group = mn.VGroup(*curve_dots)
        all_arrows_group = mn.VGroup(*velocity_vectors)

        def update_curve_dots(group: mn.Mobject) -> None:
            for dot, curve in zip(group, curves):
                dot.move_to(curve.point_from_proportion(tau.get_value()))

        def update_all_arrows(_: mn.Mobject) -> None:
            for arrow, vels, curve in zip(velocity_vectors, data.velocities, curves):
                T = vels.shape[0]
                t = min(int(tau.get_value() * (T - 1)), T - 1)
                p = curve.point_from_proportion(t / (T - 1))
                v = vec_scale * vels[t]

                if np.linalg.norm(v) < 1e-6:
                    arrow.set_opacity(0)
                else:
                    arrow.set_opacity(1)
                    arrow.put_start_and_end_on(p, p + v)

        curve_dots_group.add_updater(update_curve_dots)
        all_arrows_group.add_updater(update_all_arrows)

        progress_line = mn.NumberLine(
            x_range=(0, 1, 0.1), length=10, include_numbers=True
        ).to_edge(mn.DOWN)
        progress_dot = mn.always_redraw(
            lambda: mn.Dot(progress_line.n2p(tau.get_value()), color=mn.YELLOW)
        )
        tau_label = mn.MathTex(r"\tau", color=mn.WHITE)
        tau_label.next_to(progress_line, mn.LEFT)

        self.play(
            mn.FadeIn(curve_dots_group, all_arrows_group),
            mn.FadeIn(progress_line, progress_dot, tau_label, shift=mn.UP),
        )
        self.play(tau.animate.set_value(1), run_time=4, rate_func=mn.linear)
        for mobj in (progress_dot, curve_dots_group, all_arrows_group):
            mobj.clear_updaters()
        self.wait()

        # ── Binning ──────────────────────────────────────────────────────────────
        # Divide phase into B equal bins
        self.next_section(skip_animations=False)

        self.remove(progress_dot, curve_dots_group, all_arrows_group)
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
        bin_lines = list[mn.Line]()
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
            mn.VGroup(*(segmented_curves_per_curve[i][b] for i in range(len(curves))))  # pyright: ignore[reportArgumentType]
            for b in range(n_bins)
        ]
        target_group = segmented_group.copy()
        for b, bg in enumerate(segments_per_bin):
            target_bin = mn.VGroup(*target_group.submobjects[b::n_bins])
            target_bin.shift(radial_shift(bg))
        target_group.scale_to_fit_width(5)  # pyright: ignore[reportUnknownMemberType]
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
        others = mn.VGroup(*[seg for seg in segmented_group if seg not in bin_group])
        self.play(others.animate.set_stroke(opacity=0.25))

        shift_vec = mn.ORIGIN - bin_group.get_center()
        phase_group = mn.VGroup(progress_line, *bin_lines, tau_label)
        self.play(
            segmented_group.animate.shift(shift_vec).scale(3, about_point=mn.ORIGIN),
            mn.FadeOut(phase_group, shift=mn.DOWN),
        )

        self.wait()

        # ─────────────────────────────────────────────────────────────────────────
