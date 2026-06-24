"""
Demonstration Loader
=======
"""
# src/pacer/datasets/loader.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from pyLASAHandwritingDataset import SinglePatternMotion
from typingkit.numpy._typed.helpers import TWO

from pacer.base import Demonstrations
from pacer.datasets import HopperDataset, LASADataSet
from pacer.datasets.corruptions import (
    DemonstrationCorrupter,
    NoisyCorruptionConfig,
    NoisyDemonstrationCorrupter,
    PerPhaseBinCorruptionPlanner,
    SegmentGaussianCorrupter,
)
from pacer.datasets.interactive import InteractiveDataSet, InteractiveFigure
from pacer.datasets.interactive.legacy import LegacyInteractiveDataSet
from pacer.datasets.interactive.plugins import (
    LASALoadPlugin,
    LoadPlugin,
    SavePlugin,
    default_plugins,
)
from pacer.utils import SEED, set_seed

## ── Demonstration Loader ─────────────────────────────────────────────────────

type DemonstrationsChoice = Literal[
    "FROM_LASA",
    "HOPPER",
    "CUSTOM_FROM_LOAD",
    "CUSTOM_FROM_LASA",
    "CUSTOM_DRAW",
    "LEGACY_CUSTOM_FROM_LOAD",
    "LEGACY_CUSTOM_FROM_LASA",
    "LEGACY_CUSTOM_DRAW",
]
type CorruptionsChoice = Literal[
    "NOISY_ACTIONS", "SEGMENT_GAUSSIAN_ACTIONS", "SEGMENT_GAUSSIAN_STATES"
]


@dataclass
class DemonstrationLoaderConfig:
    choice: DemonstrationsChoice = "FROM_LASA"
    _: KW_ONLY
    LASA_pattern: SinglePatternMotion | None = None
    filepath: str | None = None
    corruptions_choice: CorruptionsChoice | None = None


@dataclass
class DemonstrationLoader:
    _: KW_ONLY
    config: DemonstrationLoaderConfig = field(default_factory=DemonstrationLoaderConfig)

    def load(self) -> Demonstrations[Any, Any, TWO, TWO]:
        demonstrations: Demonstrations[Any, Any, TWO, TWO]
        match self.config.choice:
            case "FROM_LASA":
                assert self.config.LASA_pattern is not None
                demonstrations = LASADataSet(
                    self.config.LASA_pattern
                ).to_demonstrations()
            case "HOPPER":
                demonstrations = HopperDataset().to_demonstrations()  # TODO: Fix types
            case "CUSTOM_FROM_LOAD":
                assert self.config.filepath is not None
                ifig = InteractiveFigure.create()
                plugins = default_plugins(ifig)
                plugins.append(LoadPlugin(Path(self.config.filepath)))
                drawer = InteractiveDataSet(ifig, plugins=plugins)
                drawer.show()
                demonstrations = drawer.to_demonstrations()
            case "CUSTOM_FROM_LASA":
                assert self.config.LASA_pattern is not None
                ifig = InteractiveFigure.create()
                plugins = default_plugins(ifig)
                plugins.append(LASALoadPlugin(self.config.LASA_pattern))
                drawer = InteractiveDataSet(ifig, plugins=plugins)
                drawer.show()
                demonstrations = drawer.to_demonstrations()
            case "CUSTOM_DRAW":
                ifig = InteractiveFigure.create()
                plugins = default_plugins(ifig)
                if self.config.filepath is not None:
                    plugins.append(SavePlugin(Path(self.config.filepath)))
                    plugins.append(LoadPlugin(Path(self.config.filepath)))
                drawer = InteractiveDataSet(ifig, plugins=plugins)
                drawer.show()
                demonstrations = drawer.to_demonstrations()
            case "LEGACY_CUSTOM_FROM_LOAD":
                assert self.config.filepath is not None
                legacy_drawer = LegacyInteractiveDataSet.load(self.config.filepath)
                demonstrations = legacy_drawer.to_demonstrations()
            case "LEGACY_CUSTOM_FROM_LASA":
                assert self.config.LASA_pattern is not None
                legacy_drawer = LegacyInteractiveDataSet.from_LASA(
                    self.config.LASA_pattern
                )
                plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
                demonstrations = legacy_drawer.to_demonstrations()
            case "LEGACY_CUSTOM_DRAW":
                legacy_drawer = LegacyInteractiveDataSet()
                plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
                if self.config.filepath is not None:
                    legacy_drawer.save(self.config.filepath)
                demonstrations = legacy_drawer.to_demonstrations()

        corrupter: DemonstrationCorrupter[Any, Any, TWO, TWO]
        match self.config.corruptions_choice:
            case "NOISY_ACTIONS":
                corrupter = NoisyDemonstrationCorrupter(
                    demonstrations,
                    config=NoisyCorruptionConfig(
                        noise_std=0.2,
                        outlier_fraction=0.2,
                        outlier_scale=5.0,
                        bias_strength=0.2,
                    ),
                )
                demonstrations = corrupter.inject_corruptions()
            case "SEGMENT_GAUSSIAN_ACTIONS" | "SEGMENT_GAUSSIAN_STATES":
                set_seed(SEED)
                target: Literal["STATE", "ACTION"]
                match self.config.corruptions_choice:
                    case "SEGMENT_GAUSSIAN_ACTIONS":
                        target = "ACTION"
                    case "SEGMENT_GAUSSIAN_STATES":
                        target = "STATE"
                planner = PerPhaseBinCorruptionPlanner(
                    demonstrations,
                    n_bins=7,
                    amplitude=5,
                    sigma_fraction=0.25,
                    normal_orientation="AWAY_FROM_CENTRE",
                    target=target,
                )
                corruptions = planner.plan()
                corrupter = SegmentGaussianCorrupter(demonstrations, corruptions)
                demonstrations = corrupter.inject_corruptions()
            case None:
                pass

        return demonstrations


## ─────────────────────────────────────────────────────────────────────────────
