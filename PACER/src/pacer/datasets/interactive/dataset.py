"""
Interactive Dataset
"""
# src/pacer/datasets/interactive/dataset.py

# pyright: reportUnknownMemberType = false
# pyright: reportUnusedParameter = false

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typingkit.core import TypedList
from typingkit.numpy._typed.helpers import TWO

from pacer.base import Action, Actions, Demonstration, Demonstrations, State, States
from pacer.datasets.interactive.base import (
    InteractiveController,
    InteractiveFigure,
    Plugin,
)
from pacer.datasets.interactive.plugins import default_plugins
from pacer.typings import NumDemos, NumPoints, npDType

## ── Interactive Dataset ──────────────────────────────────────────────────────


class InteractiveDataSet:
    def __init__(
        self,
        ifig: InteractiveFigure,
        *,
        plugins: list[Plugin],
        canvas_size: tuple[float, float] = (1.0, 1.0),
        min_points_to_accept: int = 5,
    ) -> None:
        self.fig: Figure
        self.ax: Axes
        self.toolbar_ax: Axes
        self.fig, self.ax, self.toolbar_ax = ifig.fig, ifig.ax, ifig.toolbar_ax

        self.ax.set_aspect("equal")

        self.controller: InteractiveController = InteractiveController(
            plugins=plugins,
            min_points=min_points_to_accept,
        )
        self.controller.on_start()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.controller.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.controller.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.controller.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.controller.on_key)

    def to_demonstrations(self) -> Demonstrations[NumDemos, NumPoints, TWO, TWO]:
        demos = self.controller.store.demos
        if not demos:
            raise RuntimeError("No demonstrations drawn.")

        out: list[Demonstration[Any, TWO, TWO]] = []

        for idx, stroke in enumerate(demos):
            arr: NDArray[np.float32] = np.asarray(stroke, dtype=npDType)
            states = [State[TWO](p) for p in arr]

            stored_vel = (
                self.controller.store.velocities[idx]
                if idx < len(self.controller.store.velocities)
                else []
            )
            if stored_vel and len(stored_vel) == len(arr):
                # Use preserved velocities (e.g. from LASA load)
                vel = np.asarray(stored_vel, dtype=npDType)
            elif len(arr) >= 2:
                diffs = np.diff(arr, axis=0)
                arc_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
                arc_lengths = np.where(arc_lengths < 1e-8, 1e-8, arc_lengths)

                unit_tangents = diffs / arc_lengths[:, None]  # direction only
                unit_tangents = np.vstack([unit_tangents, unit_tangents[[-1]]])

                # Scale to match LASA's characteristic speed
                ref_speed = self.controller.store.reference_mean_speed
                vel = np.asarray(unit_tangents * ref_speed, dtype=npDType)
            else:
                vel = np.zeros((len(arr), 2), dtype=npDType)

            actions = [Action[TWO](v) for v in vel]

            out.append(
                Demonstration(
                    index=idx,
                    states=States(states),
                    actions=Actions(actions),
                )
            )

        return Demonstrations(TypedList(out))

    def show(self) -> None:
        plt.show(block=True)

    @classmethod
    def draw(cls, **kwargs: Any) -> Self:
        ifig = InteractiveFigure.create()
        plugins = default_plugins(ifig)
        drawer = cls(plugins=plugins, **kwargs)
        drawer.show()
        return drawer

    def save(self, filepath: str) -> None:
        demos = self.controller.store.demos
        if not demos:
            print("Nothing drawn — nothing saved.")
            return

        velocities = self.controller.store.velocities
        demos_array = np.array(
            [np.asarray(d, dtype=npDType) for d in demos], dtype=object
        )
        velocities_array = np.array(
            [np.asarray(v, dtype=npDType) for v in velocities], dtype=object
        )

        np.savez_compressed(
            filepath,
            demos=demos_array,
            velocities=velocities_array,
            reference_mean_speed=self.controller.store.reference_mean_speed,
        )
        filepath = f"{filepath}.npz" if not filepath.endswith(".npz") else filepath
        print(f"Saved {len(demos)} demos to {filepath}")


## ─────────────────────────────────────────────────────────────────────────────
