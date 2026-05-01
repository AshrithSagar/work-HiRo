# tests/test_draw_interactive.py

import matplotlib.pyplot as plt

from pacer.datasets import LegacyInteractiveDataSet

drawer = LegacyInteractiveDataSet()
plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
drawer.save("hand_drawn_demos.npz")

demonstrations = LegacyInteractiveDataSet.load(
    "hand_drawn_demos.npz"
).to_demonstrations()
