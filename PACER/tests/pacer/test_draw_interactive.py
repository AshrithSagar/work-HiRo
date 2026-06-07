# tests/pacer/test_draw_interactive.py

import matplotlib.pyplot as plt
from pyLASAHandwritingDataset import SinglePatternMotion

from pacer.datasets import InteractiveDataSet, LegacyInteractiveDataSet
from pacer.datasets.interactive.base import InteractiveFigure
from pacer.datasets.interactive.plugins import LASALoadPlugin, default_plugins


def test_legacy_interactive_dataset() -> None:
    drawer = LegacyInteractiveDataSet()
    plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
    drawer.save("hand_drawn_demos.npz")

    _demonstrations = LegacyInteractiveDataSet.load(
        "hand_drawn_demos.npz"
    ).to_demonstrations()


def test_interactive_dataset(
    LASA_pattern: SinglePatternMotion = "GShape", filepath: str | None = None
) -> None:
    ifig = InteractiveFigure.create()
    plugins = default_plugins(ifig)
    plugins.append(LASALoadPlugin(LASA_pattern))
    drawer = InteractiveDataSet(ifig, plugins=plugins)
    drawer.show()
    if filepath is not None:
        drawer.save(filepath)
    _demonstrations = drawer.to_demonstrations()


if __name__ == "__main__":
    # test_legacy_interactive_dataset()

    test_interactive_dataset(
        LASA_pattern="GShape",
        filepath="hand_drawn_demos.npz",
    )
