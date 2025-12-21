"""
Plotting utils
=======
src/tp_gpt/plotting.py
"""

from typing import Any, Callable, Iterable, Optional, Protocol, cast, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from typed_numpy._typed.helpers import Array2

from tp_gpt.obstacle import CircularObstacle

OnUpdateCallback = Callable[[], Any]


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
        on_release_callback: OnUpdateCallback,
        *,
        autoscale_initial: bool = True,
        autoscale_on_release: bool = True,
        render_during_drag: bool = False,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.draggables = draggables
        self.on_release_callback = on_release_callback
        self.render_during_drag = render_during_drag
        self.autoscale_on_release = autoscale_on_release

        self.active: Optional[Draggable] = None  # The currently dragged obstacle
        self.cid_press = fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_move = fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.cid_release = fig.canvas.mpl_connect(
            "button_release_event", self._on_release
        )
        if autoscale_initial:
            self._autoscale_if_needed()

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
            if self.render_during_drag:
                self.on_release_callback()

    def _on_release(self, event: Event) -> None:
        event = cast(MouseEvent, event)
        if self.active:
            self.active.end_drag()
            self.active = None
            self.on_release_callback()
            self._autoscale_if_needed()

    def _autoscale_if_needed(self) -> None:
        if self.autoscale_on_release:
            self.ax.relim()
            self.ax.autoscale_view()
        self.fig.canvas.draw_idle()


class InteractiveCircularObstacle(CircularObstacle):
    dragging: bool = False

    def plot(self, ax: Axes, **kwargs):
        self.patch = Circle((self.center[0], self.center[1]), self.radius, **kwargs)
        ax.add_patch(self.patch)

    def contains(self, event: MouseEvent):
        if event.xdata is None or event.ydata is None:
            return False
        dx = event.xdata - self.center[0]
        dy = event.ydata - self.center[1]
        return np.hypot(dx, dy) < self.radius

    def start_drag(self, event: MouseEvent):
        self.dragging = True
        self._offset = Array2(
            [event.xdata - self.center[0], event.ydata - self.center[1]]
        )

    def drag(self, event: MouseEvent):
        if not self.dragging:
            return
        new_center = Array2(
            [event.xdata - self._offset[0], event.ydata - self._offset[1]]
        )
        self._center = new_center
        self.patch.center = tuple(new_center)

    def end_drag(self):
        self.dragging = False


class PlotSession:
    """Helper class to manage a plot session."""

    def __init__(
        self,
        *,
        figsize=(8, 8),
        title: str | None = None,
        legend: bool = True,
        tight_layout: bool = True,
        equal: bool = True,
        autoscale: bool = True,
        render_during_drag: bool = False,
    ):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        if title:
            self.ax.set_title(title)
        if equal:
            self.ax.set_aspect("equal")

        self.show_legend = legend
        self.tight_layout = tight_layout
        self.autoscale = autoscale
        self.render_during_drag = render_during_drag

    def make_lines(
        self, n: int, colormap: Colormap | str | None = None, **kwargs
    ) -> list[Line2D]:
        colors = plt.get_cmap(colormap)(np.linspace(0, 1, n))
        return [self.ax.plot([], [], color=colors[i], **kwargs)[0] for i in range(n)]

    def enable_interaction(
        self,
        draggables: Iterable[Draggable],
        on_update: OnUpdateCallback,
        *,
        initial_update: bool = True,
    ):
        if initial_update:
            on_update()
        self.interaction = InteractionManager(
            fig=self.fig,
            ax=self.ax,
            draggables=draggables,
            on_release_callback=on_update,
            autoscale_initial=True,
            autoscale_on_release=self.autoscale,
            render_during_drag=self.render_during_drag,
        )

    def show(self, show_immediately: bool = True) -> None:
        if self.show_legend:
            self.ax.legend()
        if self.tight_layout:
            self.fig.tight_layout()
        self.fig.show()
        if show_immediately:
            plt.show()


def set_axes_equal(ax: Axes3D):
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim([z_middle - max_range / 2, z_middle + max_range / 2])
