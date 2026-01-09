"""
Demo3D Viser
=======
src/tp_gpt/demo3D_viser_2.py
- Space warp
"""

import time
from typing import TypeAlias, cast

import numpy as np
import viser
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from typed_numpy._typed import TypedNDArray

from tp_gpt.core.transportation import PolicyTransportation3D
from tp_gpt.core.typings import (
    DimSpace,
    NumPoints,
    Point,
    PointSet,
    ScalarArray,
    ThreeD,
    TwoD,
)
from tp_gpt.curve import Curve3D
from tp_gpt.obstacle import SphericalObstacle
from tp_gpt.transforms import AffineTransform3D, GaussianProcessTransform3D
from tp_gpt.warp import ObstacleAvoidanceWarp3D

PointSetStack: TypeAlias = TypedNDArray[tuple[NumPoints, TwoD, DimSpace]]


def make_demo_curve_3D(n_points: int = 200) -> Curve3D:
    ts = np.linspace(0.0, 1.0, n_points)
    xs = 2.0 * ts**3 + ts**2
    ys = ts
    zs = 0.5 * np.sin(2 * np.pi * ts)
    return Curve3D.from_xyz(xs, ys, zs)


def make_policy_transportation_3D() -> PolicyTransportation3D:
    aff = AffineTransform3D(scale=False, rotate=True)
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )
    gp = GaussianProcessTransform3D(kernel=kernel, alpha=1e-10, optimizer=None)
    return PolicyTransportation3D(gp, aff)  # type: ignore


def make_grid_lines_3d(
    xlim: tuple[float, float] = (0, 4),
    ylim: tuple[float, float] = (0, 2),
    zlim: tuple[float, float] = (-1, 1),
    nx: int = 10,
    ny: int = 10,
    nz: int = 10,
) -> list[PointSet[int, ThreeD]]:
    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    zs = np.linspace(*zlim, nz)
    lines = list[PointSet[int, ThreeD]]()
    # X-aligned lines
    for y in ys:
        for z in zs:
            line = PointSet[int, ThreeD](
                np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, z)])
            )
            lines.append(line)
    # Y-aligned lines
    for x in xs:
        for z in zs:
            line = PointSet[int, ThreeD](
                np.column_stack([np.full_like(ys, x), ys, np.full_like(ys, z)])
            )
            lines.append(line)
    # Z-aligned lines
    for x in xs:
        for y in ys:
            line = PointSet[int, ThreeD](
                np.column_stack([np.full_like(zs, x), np.full_like(zs, y), zs])
            )
            lines.append(line)
    return lines


def build_warped_segments(
    lines: list[PointSet[NumPoints, ThreeD]],
    warped_lines: list[PointSet[NumPoints, ThreeD]],
) -> tuple[PointSetStack[NumPoints, ThreeD], PointSetStack[NumPoints, ThreeD]]:
    disp = ScalarArray[NumPoints](
        np.linalg.norm(np.vstack(warped_lines) - np.vstack(lines), axis=1)
    )
    dmin, dmax = float(disp.min()), float(disp.max())
    idx = int(0)
    _segments = list[list[Point[ThreeD]]]()
    _colors = list[list[tuple[int, int, int]]]()
    for line, warped in zip(lines, warped_lines):
        for i in range(len(line) - 1):
            p0, p1 = warped[i], warped[i + 1]
            color = displacement_to_color(disp[idx], dmin, dmax)
            _segments.append([p0, p1])
            _colors.append([color, color])
            idx += 1
    segments = PointSetStack[NumPoints, ThreeD](_segments)
    colors = PointSetStack[NumPoints, ThreeD](_colors)
    return segments, colors


def displacement_to_color(d: float, dmin: float, dmax: float) -> tuple[int, int, int]:
    t = np.clip((d - dmin) / (dmax - dmin + 1e-9), 0.0, 1.0)
    return (int(255 * t), int(80 * (1 - t)), int(255 * (1 - t)))


def main() -> None:
    curve = make_demo_curve_3D()
    transport = make_policy_transportation_3D()
    grid_lines = make_grid_lines_3d()

    server = viser.ViserServer()
    server.scene.add_grid(
        "/grid_infinite",
        width=1.0,
        height=1.0,
        infinite_grid=True,
        fade_distance=6.0,
        fade_strength=0.6,
    )
    server.scene.add_spline_catmull_rom(
        "/curve_source", points=curve.points, line_width=5.0, color=(180, 180, 180)
    )

    obstacles = list[SphericalObstacle]()
    obstacle_ctrls = list[viser.TransformControlsHandle]()
    endpt_ctrls = list[viser.TransformControlsHandle]()

    # GUI
    with server.gui.add_folder("Scene"):
        add_obstacle_btn = server.gui.add_button("Add obstacle")
        add_endpt_btn = server.gui.add_button("Add endpoint")
        compute_btn = server.gui.add_button("Compute space warp")
    status = server.gui.add_text("Status", initial_value="Ready", disabled=True)

    # Helpers
    def sync_obstacles() -> None:
        for obs, ctrl in zip(obstacles, obstacle_ctrls):
            obs._center = Point[ThreeD](ctrl.position)

    def clear_warped() -> None:
        try:
            server.scene.remove_by_name("/curve_warped")
            server.scene.remove_by_name("/grid_warped")
        except Exception:
            pass

    def compute_space_warp() -> None:
        try:
            sync_obstacles()
            if not endpt_ctrls:
                raise RuntimeError("Add an endpoint first")

            status.value = "Computing warp..."
            clear_warped()

            warper = ObstacleAvoidanceWarp3D(
                transportation=transport, obstacles=obstacles, curve=curve
            )
            end_pt = Point[ThreeD](endpt_ctrls[0].position)
            warper.fit(end_pt)
            warped_curve = cast(Curve3D, warper.warp_curve())
            server.scene.add_spline_catmull_rom(
                "/curve_warped",
                points=warped_curve.points,
                line_width=6.0,
                color=(255, 150, 100),
            )

            warped_lines = list[PointSet[int, ThreeD]]()
            disps = list[float]()
            for line in grid_lines:
                warped = transport.transport_positions(PointSet[int, ThreeD](line))
                warped_lines.append(warped)
                disps.append(np.linalg.norm(warped - line, axis=1).mean())

            segments, colors = build_warped_segments(grid_lines, warped_lines)
            server.scene.add_line_segments(
                "/grid_warped", points=segments, colors=colors, line_width=0.4
            )

            status.value = "Warp computed"
        except Exception as e:
            status.value = f"Error: {str(e)}"
            raise

    # UI callbacks
    @add_obstacle_btn.on_click
    def _(_):
        idx = len(obstacles)
        center = curve.points[len(curve.points) // 2] + np.array([0.3, 0.0, 0.0])
        obs = SphericalObstacle(center=center, radius=0.15, n_theta=8, n_phi=8)
        ctrl = server.scene.add_transform_controls(
            f"/obstacle_{idx}", position=center, scale=0.5, opacity=0.4
        )
        server.scene.add_icosphere(f"/obstacle_{idx}/sphere", radius=obs.radius)
        obstacles.append(obs)
        obstacle_ctrls.append(ctrl)

    @add_endpt_btn.on_click
    def _(_):
        pos = curve.end_pt + np.array([0.4, 0.0, 0.0])
        ctrl = server.scene.add_transform_controls(
            f"/endpoint_{len(endpt_ctrls)}", position=pos, scale=0.25
        )
        endpt_ctrls.append(ctrl)

    @compute_btn.on_click
    def _(_):
        compute_space_warp()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
