# tests/test_plots.py

# pyright: standard

from typing import Any, Literal

import matplotlib.pyplot as plt
from typingkit.core import RuntimeOptions, set_global_default_runtime_options
from typingkit.numpy._typed.helpers import TWO

from pacer.base import Demonstrations
from pacer.interactive import InteractiveDataSet
from pacer.lasa import LASADataSet
from pacer.plotting import plot_states_and_actions

set_global_default_runtime_options(RuntimeOptions(validate=True))


def get_demonstrations(
    choice: Literal[
        "FROM_LASA", "CUSTOM_FROM_LOAD", "CUSTOM_FROM_LASA", "CUSTOM_DRAW"
    ] = "FROM_LASA",
) -> Demonstrations[Any, Any, TWO, TWO]:
    match choice:
        case "FROM_LASA":
            return LASADataSet("GShape").to_demonstrations()
        case "CUSTOM_FROM_LOAD":
            drawer = InteractiveDataSet.load("hand_drawn_demos.npz")
            return drawer.to_demonstrations()
        case "CUSTOM_FROM_LASA":
            drawer = InteractiveDataSet.from_LASA("GShape")
            plt.show(block=True)
            return drawer.to_demonstrations()
        case "CUSTOM_DRAW":
            drawer = InteractiveDataSet()
            plt.show(block=True)
            return drawer.to_demonstrations()
    raise ValueError


if __name__ == "__main__":
    demonstrations = get_demonstrations(choice="FROM_LASA")
    plot_states_and_actions(
        demonstrations,
        demo_indices=None,
        action_scale=0.5,
        action_step=50,
    )
    plt.show()
