"""
Interactive dataset drawing
=======
Draw your own custom 2D trajectories with the mouse
"""
# src/pacer/datasets/interactive.py

# pyright: standard

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Any, Generic, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.lines import Line2D
from pyLASAHandwritingDataset import SinglePatternMotion
from typingkit.core import TypedList
from typingkit.numpy._typed.helpers import TWO, Array3D

from pacer import console
from pacer.base import Action, Actions, Demonstration, Demonstrations, State, States
from pacer.datasets.lasa import THOUSAND, LASADataSet
from pacer.typings import NumDemos, NumPoints, npDType

## ── Interactive Drawer ───────────────────────────────────────────────────────


class InteractiveDataSet(Generic[NumDemos, NumPoints]):
    """Draw demonstrations with mouse."""

    def __init__(
        self,
        canvas_size: tuple[float, float] = (1.0, 1.0),
        min_points_to_accept: int = 5,
        _suppress_welcome: bool = False,
    ) -> None:
        self.canvas_size = canvas_size
        self.min_points_to_accept = min_points_to_accept

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.ax.set_xlim(0, canvas_size[0])
        self.ax.set_ylim(0, canvas_size[1])
        self.ax.set_aspect("equal")

        if not _suppress_welcome:
            console.print(
                "PACER – Draw your own demonstrations\n"
                "------------------------------------\n"
                "Left-click + drag     = draw a trajectory (one demonstration)\n"
                "Release mouse         = finish current demonstration\n"
                "n                     = commit current stroke (optional)\n"
                "u                     = undo (remove last completed demonstration)\n"
                "r                     = reset everything\n"
                "q or close window     = finish and return dataset\n"
            )
        self.ax.set_title("Draw your demonstrations here")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.demos: list[list[tuple[float, float]]] = []
        self.demo_lines: list[Line2D] = []
        self.current_stroke: list[tuple[float, float]] = []
        self.current_artist: Line2D | None = None
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))  # type: ignore[attr-defined]

        self.finished = False

        # Event connections
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)  # type: ignore[arg-type]
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)  # type: ignore[arg-type]
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)  # type: ignore[arg-type]
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)  # type: ignore[arg-type]

    # ── Mouse handlers ───────────────────────────────────────────────────────
    def _on_press(self, event: MouseEvent) -> None:
        if not event.inaxes or self.finished:
            return
        assert event.xdata is not None and event.ydata is not None
        self.current_stroke = [(event.xdata, event.ydata)]
        color = self.colors[len(self.demos) % len(self.colors)]
        (self.current_artist,) = self.ax.plot([], [], lw=2.2, color=color)
        self.fig.canvas.draw_idle()

    def _on_motion(self, event: MouseEvent) -> None:
        if not event.inaxes or not self.current_stroke or self.finished:
            return
        assert event.xdata is not None and event.ydata is not None
        self.current_stroke.append((event.xdata, event.ydata))
        x, y = zip(*self.current_stroke)
        assert self.current_artist is not None
        self.current_artist.set_data(x, y)
        self.fig.canvas.draw_idle()

    def _on_release(self, event: MouseEvent) -> None:
        if not self.current_stroke or self.finished:
            return

        if len(self.current_stroke) >= self.min_points_to_accept:
            self.demos.append(self.current_stroke)
            assert self.current_artist is not None
            self.demo_lines.append(self.current_artist)
            console.print(
                f"Demo {len(self.demos)} accepted ({len(self.current_stroke)} points)"
            )
        else:
            console.print(f"Ignored short stroke ({len(self.current_stroke)} points)")
            # Remove the tiny/invalid stroke
            if self.current_artist is not None:
                self.current_artist.remove()
                self.fig.canvas.draw_idle()

        self.current_stroke = []
        self.current_artist = None

    def _on_key(self, event: KeyEvent) -> None:
        if event.key == "q":
            self.finished = True
            plt.close(self.fig)
            console.print(
                f"\nFinished drawing — {len(self.demos)} demonstrations created."
            )
        elif event.key == "u":
            if self.demos and self.demo_lines:
                self.demos.pop()
                last_line = self.demo_lines.pop()
                last_line.remove()
                self.fig.canvas.draw_idle()
                console.print(f"Undid last demo — now {len(self.demos)} left")
            elif self.demos:
                console.print("No line to remove (inconsistent state)")
            else:
                console.print("Nothing to undo.")
        elif event.key == "r":
            self.demos.clear()
            self.demo_lines.clear()
            self.current_stroke.clear()
            if self.current_artist is not None:
                self.current_artist.remove()
                self.current_artist = None
            self.ax.cla()
            self.ax.set_xlim(0, self.canvas_size[0])
            self.ax.set_ylim(0, self.canvas_size[1])
            self.ax.set_aspect("equal")
            self.ax.set_title("Canvas reset — start drawing again")
            self.fig.canvas.draw_idle()
            console.print("Canvas reset")
        elif event.key == "n":
            if self.current_stroke:
                # Commit current stroke
                self._on_release(None)  # type: ignore[arg-type]

    # ── Build dataset ────────────────────────────────────────────────────────
    def to_demonstrations(self) -> Demonstrations[NumDemos, NumPoints, TWO, TWO]:
        if not self.demos:
            raise RuntimeError("No demonstrations were drawn.")

        demo_list: list[Demonstration[Any, TWO, TWO]] = []
        for idx, raw_points in enumerate(self.demos):
            arr = np.array(raw_points, dtype=npDType)  # (T_i, 2)

            states_list = [State[TWO](p) for p in arr]

            if len(arr) >= 2:
                vel = np.diff(arr, axis=0)
                vel = np.vstack([vel, np.zeros((1, arr.shape[1]), dtype=npDType)])
            else:
                vel = np.zeros((len(arr), arr.shape[1]), dtype=npDType)

            actions_list = [Action[TWO](v) for v in vel]

            demo = Demonstration[NumPoints, TWO, TWO](
                index=idx,
                states=States[NumPoints, TWO](states_list),
                actions=Actions[NumPoints, TWO](actions_list),
            )
            demo_list.append(demo)

        return Demonstrations[NumDemos, NumPoints, TWO, TWO](
            TypedList[NumDemos, Demonstration[NumPoints, TWO, TWO]](demo_list)
        )

    # ── Save / Load ──────────────────────────────────────
    def save(self, filepath: str) -> None:
        if not self.demos:
            console.print("Nothing drawn — nothing saved.")
            return
        raw_arrays = [np.array(pts, dtype=npDType) for pts in self.demos]
        ragged_array = np.array(raw_arrays, dtype=object)
        np.savez_compressed(filepath, demos=ragged_array)
        filepath = f"{filepath}.npz" if not filepath.endswith(".npz") else filepath
        console.print(f"Saved {len(self.demos)} demos to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> Self:
        data = np.load(filepath, allow_pickle=True)
        ragged_array = data["demos"]
        raw_list = ragged_array.tolist()
        ds = cls(_suppress_welcome=True)
        ds.demos = [list(map(tuple, arr)) for arr in raw_list]
        console.print(f"Loaded {len(ds.demos)} raw demonstrations from {filepath}")
        return ds

    @classmethod
    def draw(cls, **kwargs: Any) -> Demonstrations[NumDemos, NumPoints, TWO, TWO]:
        """Convenience: draw -> return Demonstrations object"""
        drawer = cls(**kwargs)
        console.print("Interactive drawing window opened.")
        plt.show(block=True)
        return drawer.to_demonstrations()

    # ── From LASA ──────────────────────────────────────
    @classmethod
    def from_LASA(
        cls,
        pattern: SinglePatternMotion,
        demo_indices: list[int] | slice | int | None = None,
        min_points_to_accept: int = 5,
    ) -> Self:
        """Load one or more LASA demonstrations into the interactive drawer."""
        lasa_ds = LASADataSet(pattern)

        demo_positions: Array3D[Any, THOUSAND, TWO, np.dtype[npDType]]
        if demo_indices is None:
            demo_positions = lasa_ds.positions
        elif isinstance(demo_indices, int):
            demo_positions = lasa_ds.positions[[demo_indices]]
        elif isinstance(demo_indices, slice):
            demo_positions = lasa_ds.positions[list(range(*demo_indices.indices(7)))]
        else:
            demo_positions = lasa_ds.positions[demo_indices]

        drawer = cls(min_points_to_accept=min_points_to_accept, _suppress_welcome=True)

        # Load the demonstrations
        for i, positions in enumerate(demo_positions):
            drawer.demos.append([(float(x), float(y)) for x, y in positions])
            color = drawer.colors[i % len(drawer.colors)]
            (line,) = drawer.ax.plot(
                positions[:, 0], positions[:, 1], lw=2.2, color=color, alpha=0.9
            )
            drawer.demo_lines.append(line)

        drawer.ax.autoscale()
        drawer.ax.set_title(f"LASA '{pattern}' loaded — continue drawing or modify")

        console.print(
            f"Loaded {len(drawer.demos)} [default]demonstration(s)[/default] from LASA '{pattern}'"
        )
        return drawer


## ─────────────────────────────────────────────────────────────────────────────
