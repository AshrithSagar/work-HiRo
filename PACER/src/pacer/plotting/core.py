"""
Plotting Core
=======
Core plotting protocols and base interfaces.
"""
# src/pacer/plotting/core.py

# pyright: reportUnknownMemberType = false

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Protocol

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pacer.plotting.styles import PlotStyle

## ── Core ─────────────────────────────────────────────────────────────────────


class Artist(Protocol):
    """Renderable matplotlib artist."""

    def draw(self, ax: Axes) -> None: ...


@dataclass(slots=True, frozen=True)
class PlotRender:
    fig: Figure
    ax: Axes


@dataclass(slots=True)
class Plot2D:
    artists: list[Artist] = field(default_factory=list)

    style: PlotStyle = field(default_factory=PlotStyle)

    def add(self, *artists: Artist) -> Plot2D:
        self.artists.extend(artists)
        return self

    def render(
        self,
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> PlotRender:

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)

        for artist in self.artists:
            artist.draw(ax)

        ax.set_title(self.style.title)

        ax.set_xlabel(self.style.xlabel)
        ax.set_ylabel(self.style.ylabel)

        if self.style.equal:
            ax.axis("equal")

        ax.margins(self.style.margins)

        if self.style.legend:
            _handles, labels = ax.get_legend_handles_labels()

            if labels:
                ax.legend()

        fig.tight_layout()

        return PlotRender(fig, ax)


## ─────────────────────────────────────────────────────────────────────────────
