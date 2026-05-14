"""
Plugins
"""
# src/pacer/datasets/interactive/plugins.py

# pyright: reportUnusedParameter = false

## ── Imports ──────────────────────────────────────────────────────────────────

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, override

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure
from matplotlib.widgets import Button, TextBox
from numpy.typing import NDArray
from pyLASAHandwritingDataset import SinglePatternMotion

from pacer.datasets.interactive.base import (
    DrawMode,
    InteractiveController,
    InteractiveFigure,
    MatplotlibRenderer,
    Mode,
    Plugin,
    Point,
    Segment,
    Stroke,
)
from pacer.datasets.lasa import LASADataSet
from pacer.typings import npDType

## ── Plugins ──────────────────────────────────────────────────────────────────


def default_plugins(ifig: InteractiveFigure) -> list[Plugin]:
    renderer = MatplotlibRenderer(ifig.ax)
    binning = BinningPlugin()
    plugins: list[Plugin] = [
        SelectionPlugin(),
        SegmentMarkerPlugin(ax=ifig.ax),
        DrawingPlugin(),
        ReplacePlugin(),
        DragStrokePlugin(),
        RedrawPlugin(renderer),
        AutoscalePlugin(ifig.ax),
        ToolbarPlugin(ifig.toolbar_ax),
        BinToolbarPlugin(ifig.toolbar2_ax, binning_plugin=binning),
        binning,
        KeyboardPlugin(ifig.fig),
        StatusPlugin(ifig.status_ax),
    ]
    return plugins


@dataclass
class DrawingPlugin(Plugin):
    draw_mode: DrawMode = DrawMode.FREEHAND
    ##
    current: Stroke = field(init=False, default_factory=list)

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.DRAW:
            return
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return

        if self.draw_mode is DrawMode.FREEHAND:
            self.current = [(float(event.xdata), float(event.ydata))]
            ctrl.current_stroke = self.current
        elif self.draw_mode is DrawMode.POLYLINE:
            if not self.current:
                # Start new polyline
                self.current = [(float(event.xdata), float(event.ydata))]
                ctrl.current_stroke = self.current
            else:
                # Add vertex
                self.current.append((float(event.xdata), float(event.ydata)))

    @override
    def on_motion(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.DRAW:
            return
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return
        if self.draw_mode is DrawMode.FREEHAND and self.current:
            self.current.append((float(event.xdata), float(event.ydata)))
        # POLYLINE: motion just moves the cursor, no append until click

    @override
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.DRAW:
            return
        if self.draw_mode is DrawMode.FREEHAND:
            if not self.current:
                return
            ctrl.snapshot()
            ctrl.add_stroke(self.current)
            ctrl.current_stroke = None
            self.current = []
        # POLYLINE: release does nothing — vertices added on press, committed on enter

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        if ctrl.mode is not Mode.DRAW:
            return
        if self.draw_mode is DrawMode.POLYLINE:
            match event.key:
                case "enter":
                    if len(self.current) >= 2:
                        ctrl.snapshot()
                        ctrl.add_stroke(self.current)
                    ctrl.current_stroke = None
                    self.current = []
                case "escape":
                    ctrl.current_stroke = None
                    self.current = []
                case _:
                    pass

    @override
    def on_cancel(self, ctrl: InteractiveController) -> None:
        self.current = []
        ctrl.current_stroke = None


@dataclass
class KeyboardPlugin(Plugin):
    fig: Figure

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        match event.key:
            case "tab":
                new_mode = Mode.SELECT if ctrl.mode is Mode.DRAW else Mode.DRAW
                ctrl.mode = new_mode
                ctrl.selected_stroke = None
                ctrl.current_stroke = None
            case "z":
                ctrl.undo()
            case "delete":
                ctrl.reset()
            case "enter":
                ctrl.finish()
            case "escape":
                ctrl.cancel()
                if ctrl.mode is Mode.REPLACE:
                    ctrl.mode = Mode.SELECT  # escape from replace goes back to select
            case "a":
                for p in ctrl.plugins:
                    if isinstance(p, AutoscalePlugin):
                        p.fit(ctrl)
            case "x":
                # Enter replace mode — must have a segment selected
                if ctrl.selected_stroke is not None and ctrl.segments:
                    ctrl.mode = Mode.REPLACE
                    ctrl.current_stroke = None
                    ctrl.segment_start_idx = None
                    ctrl.segment_end_idx = None
            case "p":
                # Toggle polyline/freehand in draw mode
                for p in ctrl.plugins:
                    if isinstance(p, DrawingPlugin):
                        p.draw_mode = (
                            DrawMode.POLYLINE
                            if p.draw_mode is DrawMode.FREEHAND
                            else DrawMode.FREEHAND
                        )
            case _:
                pass

    @override
    def on_finish(self, ctrl: InteractiveController) -> None:
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.destroy()


@dataclass
class SelectionPlugin(Plugin):
    threshold: float = 0.02
    # Threshold as 2% of the diagonal of the current view

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode not in {Mode.SELECT, Mode.REPLACE}:
            return
        if event.xdata is None or event.ydata is None or not event.inaxes:
            return

        x0, x1 = event.inaxes.get_xlim()
        y0, y1 = event.inaxes.get_ylim()
        threshold = self.threshold * math.hypot(x1 - x0, y1 - y0)
        best_idx = None
        best_dist = float("inf")

        if ctrl.segments:
            for i, segment in enumerate(ctrl.segments):
                for x, y in segment.points:
                    d = math.hypot(x - event.xdata, y - event.ydata)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
        else:
            for i, stroke in enumerate(ctrl.store.demos):
                for x, y in stroke:
                    d = math.hypot(x - event.xdata, y - event.ydata)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

        ctrl.selected_stroke = best_idx if best_dist < threshold else None


@dataclass
class DragStrokePlugin(Plugin):
    ##
    dragging: bool = field(init=False, default=False)
    last: Point | None = field(init=False, default=None)

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.SELECT:
            return
        if ctrl.selected_stroke is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        ctrl.snapshot()
        self.dragging = True
        self.last = (event.xdata, event.ydata)

    @override
    def on_motion(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if not self.dragging or ctrl.selected_stroke is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        assert self.last is not None

        dx = event.xdata - self.last[0]
        dy = event.ydata - self.last[1]

        # selected_stroke is a segment index — resolve to the trajectory
        if ctrl.segments:
            seg = ctrl.segments[ctrl.selected_stroke]
            stroke = ctrl.store.demos[seg.trajectory_id]
        else:
            # fallback: no segments yet, treat as demo index
            if ctrl.selected_stroke >= len(ctrl.store.demos):
                return
            stroke = ctrl.store.demos[ctrl.selected_stroke]

        for i, (x, y) in enumerate(stroke):
            stroke[i] = (x + dx, y + dy)
        self.last = (event.xdata, event.ydata)

    @override
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        self.dragging = False


@dataclass
class RedrawPlugin(Plugin):
    renderer: MatplotlibRenderer

    def _refresh(self, ctrl: InteractiveController) -> None:
        self.renderer.reset()
        if ctrl.segments:
            # Post-binning: render by segment
            for i, segment in enumerate(ctrl.segments):
                if i == ctrl.selected_stroke:
                    # Draw halo + colored line for selected
                    self.renderer.draw(segment.points, color="gold", lw=6.0, alpha=0.5)
                    self.renderer.draw(segment.points, color="red", lw=2.2)
                else:
                    # Color by trajectory so segments of same demo share a color
                    color = f"C{segment.trajectory_id % 10}"
                    self.renderer.draw(segment.points, color=color)
        else:
            # Render whole demos
            for i, stroke in enumerate(ctrl.store.demos):
                color = "red" if i == ctrl.selected_stroke else f"C{i % 10}"
                self.renderer.draw(stroke, color=color)
        if ctrl.current_stroke is not None:
            self.renderer.draw(ctrl.current_stroke, color="black")
        self.renderer.ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    @override
    def on_start(self, ctrl: InteractiveController) -> None:
        self._refresh(ctrl)

    @override
    def on_reset(self, ctrl: InteractiveController) -> None:
        self._refresh(ctrl)

    @override
    def on_undo(self, ctrl: InteractiveController) -> None:
        self._refresh(ctrl)

    @override
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None:
        self._refresh(ctrl)

    @override
    def on_motion(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        self._refresh(ctrl)

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        self._refresh(ctrl)

    @override
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        self._refresh(ctrl)

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        self._refresh(ctrl)


@dataclass
class ToolbarPlugin(Plugin):
    toolbar_ax: Axes
    ##
    # Button state
    _btn_draw: Button = field(init=False)
    _btn_select: Button = field(init=False)
    _btn_undo: Button = field(init=False)
    _btn_reset: Button = field(init=False)
    _btn_done: Button = field(init=False)
    _btn_replace: Button = field(init=False)

    def __post_init__(self) -> None:
        ax = self.toolbar_ax
        ax.set_axis_off()

        # Divide toolbar into 5 equal slots
        positions = [0.01, 0.17, 0.33, 0.49, 0.65, 0.81]
        width = 0.17

        def _make(pos: float, label: str, callback: Callable[[Any], None]) -> Button:
            btn_ax = ax.inset_axes(bounds=(pos, 0.1, width, 0.8))  # pyright: ignore[reportUnknownMemberType]
            btn = Button(btn_ax, label)
            btn.on_clicked(callback)
            return btn

        self._btn_draw = _make(positions[0], "Draw [tab]", lambda _: None)
        self._btn_select = _make(positions[1], "Select [tab]", lambda _: None)
        self._btn_undo = _make(positions[2], "Undo [z]", lambda _: None)
        self._btn_reset = _make(positions[3], "Reset [del]", lambda _: None)
        self._btn_done = _make(positions[4], "Done [enter]", lambda _: None)
        self._btn_replace = _make(positions[5], "Replace [x]", lambda _: None)
        # Callbacks wired in on_start once we have ctrl

    @override
    def on_start(self, ctrl: InteractiveController) -> None:
        self._btn_draw.on_clicked(lambda _: self._set_mode(ctrl, Mode.DRAW))
        self._btn_select.on_clicked(lambda _: self._set_mode(ctrl, Mode.SELECT))
        self._btn_undo.on_clicked(lambda _: ctrl.undo())
        self._btn_reset.on_clicked(lambda _: ctrl.reset())
        self._btn_done.on_clicked(lambda _: ctrl.finish())
        self._btn_replace.on_clicked(lambda _: self._enter_replace(ctrl))

        self._update_mode_highlight(ctrl)

    def _set_mode(self, ctrl: InteractiveController, mode: Mode) -> None:
        ctrl.cancel()
        ctrl.mode = mode
        self._update_mode_highlight(ctrl)

    def _update_mode_highlight(self, ctrl: InteractiveController) -> None:
        mode = ctrl.mode
        self._btn_draw.ax.set_facecolor("lightblue" if mode is Mode.DRAW else "white")
        self._btn_select.ax.set_facecolor(
            "lightblue" if mode is Mode.SELECT else "white"
        )
        if mode is Mode.REPLACE:
            self._btn_select.ax.set_facecolor("orange")
        self._btn_replace.ax.set_facecolor(
            "orange" if ctrl.mode is Mode.REPLACE else "white"
        )
        self._btn_draw.ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    def _enter_replace(self, ctrl: InteractiveController) -> None:
        if ctrl.selected_stroke is not None and ctrl.segments:
            ctrl.mode = Mode.REPLACE
            ctrl.segment_start_idx = None
            ctrl.segment_end_idx = None
            self._update_mode_highlight(ctrl)

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        self._update_mode_highlight(ctrl)


@dataclass
class BinToolbarPlugin(Plugin):
    toolbar2_ax: Axes
    binning_plugin: BinningPlugin
    ##
    _textbox: TextBox = field(init=False)
    _btn_bin: Button = field(init=False)

    def __post_init__(self) -> None:
        ax = self.toolbar2_ax
        ax.set_axis_off()

        # Label
        ax.text(  # pyright: ignore[reportUnknownMemberType]
            0.01,
            0.5,
            "n_bins:",
            transform=ax.transAxes,
            va="center",
            fontsize=9,
        )

        # TextBox
        tb_ax = ax.inset_axes((0.09, 0.1, 0.12, 0.8))  # pyright: ignore[reportUnknownMemberType]
        self._textbox = TextBox(tb_ax, "", initial=str(self.binning_plugin.num_bins))

        # Button
        btn_ax = ax.inset_axes((0.23, 0.1, 0.15, 0.8))  # pyright: ignore[reportUnknownMemberType]
        self._btn_bin = Button(btn_ax, "Bin [b]")
        # Callbacks wired in on_start

    @override
    def on_start(self, ctrl: InteractiveController) -> None:
        def _on_submit(text: str) -> None:
            try:
                self.binning_plugin.num_bins = int(text)
            except ValueError:
                pass

        def _on_bin(_: Any) -> None:
            self.binning_plugin.bin_demos(ctrl)

        self._textbox.on_submit(_on_submit)
        self._textbox.on_text_change(_on_submit)  # update live as user types
        self._btn_bin.on_clicked(_on_bin)


@dataclass
class StatusPlugin(Plugin):
    status_ax: Axes
    ##
    _text: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.status_ax.set_axis_off()
        self._text = self.status_ax.text(  # pyright: ignore[reportUnknownMemberType]
            0.01,
            0.5,
            "",
            transform=self.status_ax.transAxes,
            va="center",
            fontsize=9,
            family="monospace",
        )

    def _status_str(self, ctrl: InteractiveController) -> str:
        drawing_plugin = next(
            (p for p in ctrl.plugins if isinstance(p, DrawingPlugin)), None
        )
        match ctrl.mode:
            case Mode.DRAW:
                dm = drawing_plugin.draw_mode.name.lower() if drawing_plugin else "?"
                return (
                    f"DRAW ({dm})  |  drag=draw  |  p=toggle polyline  "
                    f"|  tab=select  |  z=undo  |  del=reset  |  enter=done"
                )
            case Mode.SELECT:
                sel = (
                    f"segment {ctrl.selected_stroke}"
                    if ctrl.selected_stroke is not None
                    else "none"
                )
                return (
                    f"SELECT  |  selected: {sel}  |  click=select  |  drag=move  "
                    f"|  x=replace  |  tab=draw  |  b=bin"
                )
            case Mode.REPLACE:
                if ctrl.selected_stroke is None:
                    return "REPLACE  |  click near a segment to select it"
                if ctrl.segment_start_idx is None:
                    return "REPLACE  |  drag along trajectory to select range"
                if ctrl.segment_end_idx is None:
                    return "REPLACE  |  drag along trajectory to select range..."
                return "REPLACE  |  now drag to draw replacement curve  |  esc=cancel"

    def _refresh(self, ctrl: InteractiveController) -> None:
        self._text.set_text(self._status_str(ctrl))
        self.status_ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    @override
    def on_start(self, ctrl: InteractiveController) -> None:
        self._refresh(ctrl)

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        self._refresh(ctrl)

    @override
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        self._refresh(ctrl)

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        self._refresh(ctrl)

    @override
    def on_reset(self, ctrl: InteractiveController) -> None:
        self._refresh(ctrl)

    @override
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None:
        self._refresh(ctrl)


@dataclass
class AutoscalePlugin(Plugin):
    ax: Axes
    margin: float = 0.05

    def fit(self, ctrl: InteractiveController) -> None:
        all_pts = [pt for stroke in ctrl.store.demos for pt in stroke]
        if not all_pts:
            return
        xs, ys = zip(*all_pts)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xpad = (xmax - xmin) * self.margin or self.margin
        ypad = (ymax - ymin) * self.margin or self.margin
        self.ax.set_xlim(xmin - xpad, xmax + xpad)
        self.ax.set_ylim(ymin - ypad, ymax + ypad)
        self.ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    @override
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None:
        self.fit(ctrl)

    @override
    def on_reset(self, ctrl: InteractiveController) -> None:
        self.fit(ctrl)

    @override
    def on_undo(self, ctrl: InteractiveController) -> None:
        self.fit(ctrl)


@dataclass
class SmoothingPlugin(Plugin):
    window: int = 5

    @override
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None:
        if len(stroke) < self.window:
            return

        arr: NDArray[np.float32] = np.asarray(stroke, dtype=npDType)
        kernel = np.ones(self.window, dtype=npDType) / self.window
        smoothed = np.vstack(
            [np.convolve(arr[:, i], kernel, mode="same") for i in range(arr.shape[1])]
        ).T
        stroke[:] = [(float(x), float(y)) for x, y in smoothed]


@dataclass
class ResamplePlugin(Plugin):
    num_points: int = 100

    @override
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None:
        if len(stroke) < 2:
            return

        arr: NDArray[np.float32] = np.asarray(stroke, dtype=npDType)
        t_old = np.linspace(0.0, 1.0, len(arr))
        t_new = np.linspace(0.0, 1.0, self.num_points)
        resampled = np.vstack(
            [np.interp(t_new, t_old, arr[:, i]) for i in range(arr.shape[1])]
        ).T
        stroke[:] = [(float(x), float(y)) for x, y in resampled]


@dataclass
class SavePlugin(Plugin):
    filepath: Path

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        if event.key != "s":
            return

        demos = [np.asarray(stroke, dtype=npDType) for stroke in ctrl.store.demos]
        np.savez_compressed(self.filepath, demos=np.array(demos, dtype=object))


@dataclass
class LoadPlugin(Plugin):
    filepath: Path
    auto: bool = True

    def _load(self, ctrl: InteractiveController) -> None:
        data = np.load(self.filepath, allow_pickle=True)
        demos = data["demos"]
        ctrl.reset()
        for stroke in demos:
            ctrl.add_stroke([(float(x), float(y)) for x, y in stroke])

    @override
    def on_start(self, ctrl: InteractiveController) -> None:
        if self.auto:
            self._load(ctrl)

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        if event.key == "o":
            self._load(ctrl)


@dataclass
class LASALoadPlugin(Plugin):
    pattern: SinglePatternMotion
    auto: bool = True
    ##
    reference_mean_speed: float = field(init=False, default=1.0)

    def _load(self, ctrl: InteractiveController) -> None:
        ctrl.store.reset()

        ds = LASADataSet(self.pattern)
        speeds: list[float] = []
        for positions, velocities in zip(ds.positions, ds.velocities):
            stroke = [(float(x), float(y)) for x, y in positions]
            vel_stroke = [(float(vx), float(vy)) for vx, vy in velocities]
            ctrl.store.add(stroke, min_points=0, velocity=vel_stroke)
            speeds.append(np.hypot(velocities[:, 0], velocities[:, 1]).mean())
        self.reference_mean_speed = float(np.mean(speeds))
        ctrl.store.reference_mean_speed = self.reference_mean_speed

        for p in ctrl.plugins:
            p.on_reset(ctrl)

    @override
    def on_start(self, ctrl: InteractiveController) -> None:
        if self.auto:
            self._load(ctrl)

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        if event.key == "l":
            self._load(ctrl)


@dataclass
class SegmentMarkerPlugin(Plugin):
    """Drag along trajectory in REPLACE mode to select start..end range."""

    ax: Axes

    ##
    _dragging: bool = field(init=False, default=False)
    _range_line: Any = field(init=False, default=None)  # highlighted range artist
    _start_dot: Any = field(init=False, default=None)
    _end_dot: Any = field(init=False, default=None)

    def _closest_idx(self, stroke: Stroke, x: float, y: float) -> int:
        arr = np.asarray(stroke, dtype=np.float32)
        dists = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
        return int(np.argmin(dists))

    def _draw_range(
        self,
        ctrl: InteractiveController,
        traj: Stroke,
        start: int,
        end: int,
        ax: Axes,
    ) -> None:
        self._clear_artists(ax)
        if start > end:
            start, end = end, start
        pts = traj[start : end + 1]
        if len(pts) >= 2:
            xs, ys = zip(*pts)
            self._range_line = ax.plot(  # pyright: ignore[reportUnknownMemberType]
                xs, ys, lw=5.0, color="yellow", alpha=0.7, zorder=4
            )[0]
        # Dot markers at endpoints
        self._start_dot = ax.plot(  # pyright: ignore[reportUnknownMemberType]
            traj[start][0], traj[start][1], "o", color="green", ms=8, zorder=5
        )[0]
        self._end_dot = ax.plot(  # pyright: ignore[reportUnknownMemberType]
            traj[end][0], traj[end][1], "o", color="red", ms=8, zorder=5
        )[0]
        ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    def _clear_artists(self, ax: Axes) -> None:
        for attr in ("_range_line", "_start_dot", "_end_dot"):
            artist = getattr(self, attr)
            if artist is not None:
                artist.remove()
                setattr(self, attr, None)
        ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.REPLACE:
            return
        if ctrl.selected_stroke is None:
            return
        # Only start range selection if endpoints not yet committed
        if ctrl.segment_start_idx is not None and ctrl.segment_end_idx is not None:
            return  # range already set, ReplacePlugin draws now
        if event.xdata is None or event.ydata is None or not event.inaxes:
            return

        seg = ctrl.segments[ctrl.selected_stroke]
        traj = ctrl.store.demos[seg.trajectory_id]
        idx = self._closest_idx(traj, event.xdata, event.ydata)
        ctrl.segment_start_idx = idx
        ctrl.segment_end_idx = None
        self._dragging = True
        self._draw_range(ctrl, traj, idx, idx, event.inaxes)

    @override
    def on_motion(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if not self._dragging or ctrl.mode is not Mode.REPLACE:
            return
        if ctrl.selected_stroke is None or ctrl.segment_start_idx is None:
            return
        if event.xdata is None or event.ydata is None or not event.inaxes:
            return

        seg = ctrl.segments[ctrl.selected_stroke]
        traj = ctrl.store.demos[seg.trajectory_id]
        cur_idx = self._closest_idx(traj, event.xdata, event.ydata)
        self._draw_range(
            ctrl,
            traj,
            ctrl.segment_start_idx,
            cur_idx,
            event.inaxes,
        )

    @override
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if not self._dragging:
            return
        self._dragging = False

        if ctrl.selected_stroke is None or ctrl.segment_start_idx is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        seg = ctrl.segments[ctrl.selected_stroke]
        traj = ctrl.store.demos[seg.trajectory_id]
        end_idx = self._closest_idx(traj, event.xdata, event.ydata)

        # Ensure ordering
        start = ctrl.segment_start_idx
        end = end_idx
        if start > end:
            start, end = end, start
        ctrl.segment_start_idx = start
        ctrl.segment_end_idx = end

    def _cancel(self, ctrl: InteractiveController) -> None:
        self._dragging = False
        # Clear visual markers — need an axes ref; use ctrl.segments if available
        if ctrl.segments and ctrl.selected_stroke is not None:
            _seg = ctrl.segments[ctrl.selected_stroke]
            # We don't have ax directly — store it
        # Simpler: just remove whatever artists exist
        for attr in ("_range_line", "_start_dot", "_end_dot"):
            artist = getattr(self, attr)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, attr, None)
        ctrl.segment_start_idx = None
        ctrl.segment_end_idx = None

    @override
    def on_cancel(self, ctrl: InteractiveController) -> None:
        self._cancel(ctrl)

    @override
    def on_reset(self, ctrl: InteractiveController) -> None:
        self._cancel(ctrl)

    @override
    def on_undo(self, ctrl: InteractiveController) -> None:
        self._cancel(ctrl)


@dataclass
class BinningPlugin(Plugin):
    num_bins: int = 10
    ##
    _has_binned: bool = field(init=False, default=False)

    def bin_demos(self, ctrl: InteractiveController) -> None:
        """Rebuild ctrl.segments by splitting demos into num_bins segments each."""
        self._has_binned = True
        ctrl.segments.clear()
        for traj_id, stroke in enumerate(ctrl.store.demos):
            if len(stroke) < self.num_bins:
                ctrl.segments.append(
                    Segment(
                        points=list(stroke),
                        trajectory_id=traj_id,
                        segment_id=0,
                    )
                )
                continue
            arr = np.asarray(stroke, dtype=npDType)
            for seg_id, seg_idx in enumerate(
                np.array_split(np.arange(len(arr)), self.num_bins)
            ):
                if len(seg_idx) == 0:
                    continue
                ctrl.segments.append(
                    Segment(
                        points=[(float(arr[i, 0]), float(arr[i, 1])) for i in seg_idx],
                        trajectory_id=traj_id,
                        segment_id=seg_id,
                    )
                )
        for p in ctrl.plugins:
            p.on_reset(ctrl)

    @override
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None:
        # Re-bin automatically only if user has already binned once
        if self._has_binned:
            self.bin_demos(ctrl)

    @override
    def on_undo(self, ctrl: InteractiveController) -> None:
        if self._has_binned:
            self.bin_demos(ctrl)

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        if event.key == "b":
            self.bin_demos(ctrl)


@dataclass
class ReplacePlugin(Plugin):
    ##
    current: Stroke = field(init=False, default_factory=list)

    def _ready_to_draw(self, ctrl: InteractiveController) -> bool:
        """Both endpoints picked, ready to accept a new drawn stroke."""
        return ctrl.segment_start_idx is not None and ctrl.segment_end_idx is not None

    def _resample_stroke(self, stroke: Stroke, n: int) -> Stroke:
        """Resample stroke to exactly n points using cumulative arc length."""
        arr = np.asarray(stroke, dtype=npDType)
        if len(arr) < 2 or n < 2:
            return stroke
        deltas = np.diff(arr, axis=0)
        seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cumlen[-1]
        if total == 0.0:
            return stroke
        t_old = cumlen / total
        t_new = np.linspace(0.0, 1.0, n)
        resampled = np.vstack([np.interp(t_new, t_old, arr[:, i]) for i in range(2)]).T
        return [(float(x), float(y)) for x, y in resampled]

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.REPLACE:
            return
        if ctrl.selected_stroke is None:
            return
        if not self._ready_to_draw(ctrl):
            return  # still picking endpoints — SegmentMarkerPlugin handles this
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return
        self.current = [(float(event.xdata), float(event.ydata))]
        ctrl.current_stroke = self.current

    @override
    def on_motion(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.REPLACE:
            return
        if not self.current:
            return
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return
        self.current.append((float(event.xdata), float(event.ydata)))
        # current_stroke already points to self.current, RedrawPlugin will pick it up

    @override
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if not self.current:
            return
        if ctrl.selected_stroke is None or not self._ready_to_draw(ctrl):
            self.current = []
            ctrl.current_stroke = None
            return

        seg = ctrl.segments[ctrl.selected_stroke]
        traj_id = seg.trajectory_id
        old_traj = ctrl.store.demos[traj_id]
        start = ctrl.segment_start_idx
        end = ctrl.segment_end_idx
        assert start is not None and end is not None

        seg_len = end - start
        if seg_len < 1 or len(self.current) < 2:
            self.current = []
            ctrl.current_stroke = None
            return

        resampled = self._resample_stroke(self.current, seg_len)
        spliced: Stroke = old_traj[:start] + resampled + old_traj[end:]

        ctrl.snapshot()
        ctrl.store.demos[traj_id] = spliced

        # Clear replace state
        ctrl.segment_start_idx = None
        ctrl.segment_end_idx = None
        ctrl.selected_stroke = None
        ctrl.current_stroke = None
        self.current = []

        # This triggers BinningPlugin.on_demo_added -> rebuilds segments -> redraws
        for p in ctrl.plugins:
            p.on_demo_added(ctrl, spliced)

    @override
    def on_cancel(self, ctrl: InteractiveController) -> None:
        self.current = []
        ctrl.current_stroke = None


## ─────────────────────────────────────────────────────────────────────────────
