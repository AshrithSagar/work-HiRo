"""
Styles
======
Styling configuration.
"""
# src/pacer/plotting/styles.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass

## ── Styles ───────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class LineStyle:
    linewidth: float = 2.0
    alpha: float = 1.0
    linestyle: str = "-"
    color: str | None = None


@dataclass(slots=True, frozen=True)
class ScatterStyle:
    size: float = 24.0
    alpha: float = 1.0
    color: str | None = None


@dataclass(slots=True, frozen=True)
class QuiverStyle:
    scale: float = 1.0
    alpha: float = 0.8
    width: float = 0.003
    color: str | None = None


@dataclass(slots=True, frozen=True)
class PlotStyle:
    figsize: tuple[float, float] = (6, 6)

    title: str = ""
    xlabel: str = "x"
    ylabel: str = "y"

    equal: bool = True
    margins: float = 0.05

    legend: bool = True


## ─────────────────────────────────────────────────────────────────────────────
