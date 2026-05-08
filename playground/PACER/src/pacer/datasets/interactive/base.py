"""
Interactive Base
"""
# src/pacer/datasets/interactive/base.py

# pyright: reportUnknownMemberType = false
# pyright: reportUnusedParameter = false

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

## ── Base ─────────────────────────────────────────────────────────────────────

type Point = tuple[float, float]
type Stroke = list[Point]


@dataclass
class Segment:
    points: Stroke
    trajectory_id: int
    segment_id: int


@dataclass
class DemoStore:
    demos: list[Stroke]

    def add(self, stroke: Stroke, *, min_points: int) -> bool:
        if len(stroke) < min_points:
            return False
        self.demos.append(stroke)
        return True

    def undo(self) -> None:
        if self.demos:
            self.demos.pop()

    def reset(self) -> None:
        self.demos.clear()


@dataclass
class InteractiveFigure:
    fig: Figure
    ax: Axes
    toolbar_ax: Axes  # First row for controls: draw, select, reset, undo.
    toolbar2_ax: Axes  # Second row for bin controls
    status_ax: Axes  # bottom status bar

    @classmethod
    def create(cls) -> Self:
        fig, (toolbar_ax, toolbar2_ax, status_ax, ax) = plt.subplots(
            4, 1, gridspec_kw={"height_ratios": [1, 0.6, 0.5, 11], "hspace": 0.05}
        )
        return cls(
            fig=fig,
            ax=ax,
            toolbar_ax=toolbar_ax,
            toolbar2_ax=toolbar2_ax,
            status_ax=status_ax,
        )


class MatplotlibRenderer:
    def __init__(self, ax: Axes) -> None:
        self.ax: Axes = ax
        self.lines: list[Line2D] = []

    def draw(
        self,
        points: Iterable[Point],
        *,
        color: str,
        lw: float = 2.2,
        alpha: float = 1.0,
    ) -> None:
        if not points:
            return
        pts = list(points)
        if len(pts) < 2:
            return
        x, y = zip(*pts)
        (line,) = self.ax.plot(x, y, lw=lw, color=color, alpha=alpha)
        self.lines.append(line)

    def reset(self) -> None:
        for line in self.lines:
            line.remove()
        self.lines.clear()


class Plugin:
    def on_start(self, ctrl: InteractiveController) -> None: ...
    def on_reset(self, ctrl: InteractiveController) -> None: ...
    def on_undo(self, ctrl: InteractiveController) -> None: ...
    def on_finish(self, ctrl: InteractiveController) -> None: ...
    def on_cancel(self, ctrl: InteractiveController) -> None: ...
    def on_press(self, ctrl: InteractiveController, event: MouseEvent) -> None: ...
    def on_motion(self, ctrl: InteractiveController, event: MouseEvent) -> None: ...
    def on_release(self, ctrl: InteractiveController, event: MouseEvent) -> None: ...
    def on_key(self, ctrl: InteractiveController, event: KeyEvent) -> None: ...
    def on_demo_added(self, ctrl: InteractiveController, stroke: Stroke) -> None: ...


class Mode(Enum):
    DRAW = auto()
    SELECT = auto()
    REPLACE = auto()


class DrawMode(Enum):
    FREEHAND = auto()
    POLYLINE = auto()


class InteractiveController:
    def __init__(
        self,
        *,
        plugins: list[Plugin],
        min_points: int = 5,
    ) -> None:
        self.store: DemoStore = DemoStore([])
        self.plugins: list[Plugin] = plugins
        self.min_points: int = min_points
        self.mode: Mode = Mode.DRAW
        self.selected_stroke: int | None = None
        self.current_stroke: Stroke | None = None
        self.segment_start_idx: int | None = None  # Point index within trajectory
        self.segment_end_idx: int | None = None
        self.segments: list[Segment] = []
        self.history: list[list[Stroke]] = []

    # ── Event dispatch  ───────────────────────────────────────────────────────

    def on_start(self) -> None:
        for p in self.plugins:
            p.on_start(self)

    def on_press(self, event: MouseEvent) -> None:
        for p in self.plugins:
            p.on_press(self, event)

    def on_motion(self, event: MouseEvent) -> None:
        for p in self.plugins:
            p.on_motion(self, event)

    def on_release(self, event: MouseEvent) -> None:
        for p in self.plugins:
            p.on_release(self, event)

    def on_key(self, event: KeyEvent) -> None:
        for p in self.plugins:
            p.on_key(self, event)

    # ── API  ──────────────────────────────────────────────────────────────────

    def add_stroke(self, stroke: Stroke) -> bool:
        ok: bool = self.store.add(stroke, min_points=self.min_points)
        if ok:
            for p in self.plugins:
                p.on_demo_added(self, stroke)
        return ok

    def snapshot(self) -> None:
        self.history.append(deepcopy(self.store.demos))

    def undo(self) -> None:
        if self.history:
            self.store.demos = self.history.pop()
            for p in self.plugins:
                p.on_undo(self)

    def finish(self) -> None:
        for p in self.plugins:
            p.on_finish(self)

    def reset(self) -> None:
        self.store.reset()
        for p in self.plugins:
            p.on_reset(self)

    def cancel(self) -> None:
        self.selected_stroke = None
        self.current_stroke = None
        self.segment_start_idx = None
        self.segment_end_idx = None
        for p in self.plugins:
            p.on_cancel(self)


## ─────────────────────────────────────────────────────────────────────────────
