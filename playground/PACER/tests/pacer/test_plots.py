# tests/test_plots.py

# pyright: standard

import matplotlib.pyplot as plt
from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer.plotting import plot_states_and_actions
from pacer.testutils import DemonstrationLoader

set_global_default_runtime_options(RuntimeOptions(validate=True))


if __name__ == "__main__":
    demonstrations = DemonstrationLoader(choice="FROM_LASA", pattern="GShape").load()
    plot_states_and_actions(
        demonstrations,
        demo_indices=None,
        action_scale=0.5,
        action_step=50,
    )
    plt.show()
