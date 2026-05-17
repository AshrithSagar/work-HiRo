"""
Visualisation utils
=======
"""
# src/pacer/visualisation.py

# pyright: reportUnknownMemberType = false

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Generic

import matplotlib.pyplot as plt
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import Demonstrations
from pacer.pacer import PACERResult
from pacer.plotting import (
    plot_action_comparison,
    plot_action_correction_magnitude,
    plot_phases,
    plot_ribbon_action_field,
    plot_state_comparison,
    plot_states_before_after,
    plot_trajectories,
    plot_trust_values,
)
from pacer.typings import NumBins, NumDemos, NumPoints

## ── Visualisation ────────────────────────────────────────────────────────────


@dataclass
class PACERVisualisationConfig:
    show: bool = True
    save_dir: Path | str | None = None

    trajectories: bool = True
    phases: bool = True
    trust_values: bool = True

    states_before_after: bool = True
    action_comparison: bool = True
    state_comparison: bool = True

    ribbon_action_field: bool = True
    action_correction_magnitude: bool = True


@dataclass
class PACERVisualiser(Generic[NumBins, NumDemos, NumPoints]):
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO]
    pacer_result: PACERResult[NumBins, NumDemos, NumPoints, TWO, TWO]
    _: KW_ONLY
    config: PACERVisualisationConfig = field(default_factory=PACERVisualisationConfig)

    def render(self) -> None:
        if self.config.trajectories:
            plot_trajectories(self.demonstrations)

        if self.config.phases:
            plot_phases(self.pacer_result.phases)

        if self.config.trust_values:
            plot_trust_values(
                self.pacer_result.action_trust_values,
                title="Action trust values",
            )

            if self.pacer_result.state_trust_values is not None:
                plot_trust_values(
                    self.pacer_result.state_trust_values,
                    title="State trust values",
                )

        if self.config.ribbon_action_field:
            plot_ribbon_action_field(self.pacer_result.bins)

        if self.config.action_comparison:
            plot_action_comparison(
                self.demonstrations[0].actions,
                self.pacer_result.pseudo_labels.actions[0],
                title="Demo 0: Action refinement",
            )

        if self.config.states_before_after:
            if self.pacer_result.pseudo_labels.states is not None:
                plot_states_before_after(
                    self.demonstrations.states,
                    self.pacer_result.pseudo_labels.states,
                )

        if self.config.state_comparison:
            if self.pacer_result.pseudo_labels.states is not None:
                plot_state_comparison(
                    self.demonstrations[0].states,
                    self.pacer_result.pseudo_labels.states[0],
                    title="Demo 0: State refinement",
                )

        if self.config.action_correction_magnitude:
            plot_action_correction_magnitude(
                self.demonstrations.actions,
                self.pacer_result.pseudo_labels.actions,
            )

        if self.config.save_dir is not None:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            fignums = plt.get_fignums()
            with console.status("Saving figures...", spinner="dots"):
                for i in fignums:
                    fig = plt.figure(i)
                    fig.savefig(
                        save_dir / f"figure_{i}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
            console.print(
                f"[grey53]Saved {len(fignums)} figures to[/grey53] "
                + f"[italic grey69]{save_dir.absolute()}[/italic grey69]"
            )

        if self.config.show:
            plt.show()


## ─────────────────────────────────────────────────────────────────────────────
