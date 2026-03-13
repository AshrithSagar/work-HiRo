# tests/test_draw_interactive.py

# pyright: standard

import matplotlib.pyplot as plt

from pacer.interactive import InteractiveDataSet

drawer = InteractiveDataSet()
plt.show(block=True)
drawer.save("hand_drawn_demos.npz")

demonstrations = InteractiveDataSet.load("hand_drawn_demos.npz").to_demonstrations()
