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
from matplotlib.widgets import Button
from numpy.typing import NDArray
from pyLASAHandwritingDataset import SinglePatternMotion

from pacer.datasets.interactive.base import (
    InteractiveController,
    InteractiveFigure,
    MatplotlibRenderer,
    Mode,
    Plugin,
    Point,
    Stroke,
)
from pacer.datasets.lasa import LASADataSet
from pacer.typings import npDType

## ── Plugins ──────────────────────────────────────────────────────────────────


def default_plugins(ifig: InteractiveFigure) -> list[Plugin]:
    renderer = MatplotlibRenderer(ifig.ax)
    plugins: list[Plugin] = [
        SelectionPlugin(),
        DrawingPlugin(),
        DragStrokePlugin(),
        RedrawPlugin(renderer),
        AutoscalePlugin(ifig.ax),
        ToolbarPlugin(ifig.toolbar_ax),
        KeyboardPlugin(ifig.fig),
    ]
    return plugins


@dataclass
class DrawingPlugin(Plugin):
    ##
    current: Stroke = field(init=False, default_factory=list)

    @override
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if ctrl.mode is not Mode.DRAW:
            return
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return

        self.current = [(float(event.xdata), float(event.ydata))]
        ctrl.current_stroke = self.current

    @override
    def on_motion(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if not event.inaxes or not self.current:
            return

        assert event.xdata is not None and event.ydata is not None
        self.current.append((float(event.xdata), float(event.ydata)))

    @override
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None:
        if not self.current:
            return

        ctrl.snapshot()
        ctrl.add_stroke(self.current)
        ctrl.current_stroke = None
        self.current = []


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
                ctrl.selected_stroke = None
                ctrl.current_stroke = None
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
        if ctrl.mode is not Mode.SELECT:
            return
        if event.xdata is None or event.ydata is None or not event.inaxes:
            return

        x0, x1 = event.inaxes.get_xlim()
        y0, y1 = event.inaxes.get_ylim()
        threshold = self.threshold * math.hypot(x1 - x0, y1 - y0)
        best_idx = None
        best_dist = float("inf")
        for i, stroke in enumerate(ctrl.store.demos):
            for x, y in stroke:
                d = math.hypot(x - event.xdata, y - event.ydata)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
        ctrl.selected_stroke = best_idx if best_dist < threshold else None


@dataclass
class DragStrokePlugin(Plugin):
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
        strokes = list(ctrl.store.demos)  # Shallow copy
        if ctrl.current_stroke is not None:
            strokes.append(ctrl.current_stroke)
        self.renderer.redraw(strokes, ctrl.selected_stroke)

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

    def __post_init__(self) -> None:
        ax = self.toolbar_ax
        ax.set_axis_off()

        # Divide toolbar into 5 equal slots
        positions = [0.01, 0.21, 0.41, 0.61, 0.81]
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
        # Callbacks wired in on_start once we have ctrl

    @override
    def on_start(self, ctrl: InteractiveController) -> None:
        self._btn_draw.on_clicked(lambda _: self._set_mode(ctrl, Mode.DRAW))
        self._btn_select.on_clicked(lambda _: self._set_mode(ctrl, Mode.SELECT))
        self._btn_undo.on_clicked(lambda _: ctrl.undo())
        self._btn_reset.on_clicked(lambda _: ctrl.reset())
        self._btn_done.on_clicked(lambda _: ctrl.finish())
        self._update_mode_highlight(ctrl.mode)

    def _set_mode(self, ctrl: InteractiveController, mode: Mode) -> None:
        ctrl.mode = mode
        ctrl.selected_stroke = None
        self._update_mode_highlight(mode)

    def _update_mode_highlight(self, mode: Mode) -> None:
        self._btn_draw.ax.set_facecolor("lightblue" if mode is Mode.DRAW else "white")
        self._btn_select.ax.set_facecolor(
            "lightblue" if mode is Mode.SELECT else "white"
        )
        self._btn_draw.ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    @override
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None:
        self._update_mode_highlight(ctrl.mode)


@dataclass
class AutoscalePlugin(Plugin):
    ax: Axes
    margin: float = 0.05

    def _rescale(self, ctrl: InteractiveController) -> None:
        all_pts = [pt for stroke in ctrl.store.demos for pt in stroke]
        if not all_pts:
            return
        xs, ys = zip(*all_pts)
        xpad = (max(xs) - min(xs)) * self.margin or self.margin
        ypad = (max(ys) - min(ys)) * self.margin or self.margin
        self.ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        self.ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
        self.ax.figure.canvas.draw_idle()  # pyright: ignore[reportUnknownMemberType]

    @override
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None:
        self._rescale(ctrl)

    @override
    def on_reset(self, ctrl: InteractiveController) -> None:
        self._rescale(ctrl)

    @override
    def on_undo(self, ctrl: InteractiveController) -> None:
        self._rescale(ctrl)


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

    def _load(self, ctrl: InteractiveController) -> None:
        ds = LASADataSet(self.pattern)
        ctrl.store.reset()
        for demo in ds.positions:
            ctrl.add_stroke([(float(x), float(y)) for x, y in demo])
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


## ─────────────────────────────────────────────────────────────────────────────
