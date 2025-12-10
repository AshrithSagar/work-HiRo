"""
Plotting utils
=========
src/tp_gpt/plotting.py
"""

from typing import Any, Callable, Iterable, Optional, Protocol, cast, runtime_checkable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from typed_numpy.helpers import Array1D

from tp_gpt.obstacle import CircularObstacle

OnReleaseCallback = Callable[[], Any]


@runtime_checkable
class Draggable(Protocol):
    """Protocol representing an object that can be dragged on a Matplotlib axis."""

    def contains(self, event: MouseEvent) -> bool: ...
    def start_drag(self, event: MouseEvent) -> None: ...
    def drag(self, event: MouseEvent) -> None: ...
    def end_drag(self) -> None: ...


class InteractionManager:
    """Manages drag events for multiple draggable objects."""

    def __init__(
        self,
        fig: Figure,
        ax: Axes,
        draggables: Iterable[Draggable],
        on_release_callback: OnReleaseCallback,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.draggables = draggables
        self.on_release_callback = on_release_callback

        self.active: Optional[Draggable] = None  # The currently dragged obstacle
        self.cid_press = fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_move = fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.cid_release = fig.canvas.mpl_connect(
            "button_release_event", self._on_release
        )

    def _on_press(self, event: Event) -> None:
        event = cast(MouseEvent, event)
        for draggable in self.draggables:
            if draggable.contains(event):
                self.active = draggable
                draggable.start_drag(event)
                break

    def _on_move(self, event: Event) -> None:
        event = cast(MouseEvent, event)
        if self.active:
            self.active.drag(event)
            self.fig.canvas.draw_idle()

    def _on_release(self, event: Event) -> None:
        event = cast(MouseEvent, event)
        if self.active:
            self.active.end_drag()
            self.active = None
            self.on_release_callback()


class InteractiveCircularObstacle(CircularObstacle):
    dragging: bool = False

    def plot(self, ax: Axes, **kwargs):
        self.patch = Circle((self.center[0], self.center[1]), self.radius, **kwargs)
        ax.add_patch(self.patch)

    def contains(self, event: MouseEvent):
        if event.xdata is None or event.ydata is None:
            return False
        return (
            np.hypot(event.xdata - self.center[0], event.ydata - self.center[1])
            < self.radius
        )

    def start_drag(self, event: MouseEvent):
        self.dragging = True
        self._offset = np.array(
            [event.xdata - self.center[0], event.ydata - self.center[1]]
        )

    def drag(self, event: MouseEvent):
        if not self.dragging:
            return
        new_center = [event.xdata - self._offset[0], event.ydata - self._offset[1]]
        self._center = Array1D(new_center)
        self.patch.center = tuple(new_center)

    def end_drag(self):
        self.dragging = False
