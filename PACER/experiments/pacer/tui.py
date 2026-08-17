"""
PACER Ablation TUI
=======
experiments/pacer/tui.py
"""
# experiments/pacer/tui.py

from __future__ import annotations

import argparse
import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any

from ablation_pacer_bc import (
    ALL_GROUP_NAMES,
    BASELINE_BC_TRAIN_CONFIG,
    GROUPS,
    collect_row,
    make_baseline_pacer_config,
)
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    ListItem,
    ListView,
    LoadingIndicator,
    Static,
)

from pacer.datasets import DemonstrationLoader, DemonstrationLoaderConfig
from pacer.experiments import BCExperiment, PACERBCExperiment
from pacer.pacer import PACERConfig

## ── Flatten the ablation groups into a single ordered list ────────────────────

CONFIG_FIELDS = [
    "phase_pipeline_config",
    "n_bins",
    "consensus_config",
    "action_trust_value_params",
    "action_pseudo_label_params",
    "use_state_labels",
    "state_trust_value_params",
    "state_pseudo_label_params",
]


@dataclass
class Item:
    group: str
    variant: str
    description: str
    mutate: Any  # Mutator, from ablation.py


def flatten_items() -> list[Item]:
    items: list[Item] = []
    for group in ALL_GROUP_NAMES:
        for variant, (mutate, description) in GROUPS[group].items():
            items.append(
                Item(
                    group=group, variant=variant, description=description, mutate=mutate
                )
            )
    return items


## ── Config diff rendering ───────────────────────────────────────────────────


def describe_config(baseline: PACERConfig, variant: PACERConfig) -> Text:
    text = Text()
    for name in CONFIG_FIELDS:
        b_repr = repr(getattr(baseline, name))
        v_repr = repr(getattr(variant, name))
        changed = b_repr != v_repr
        marker = "\u25b6 " if changed else "  "
        text.append(
            f"{marker}{name}\n", style="bold yellow" if changed else "bold grey50"
        )
        text.append(f"    {v_repr}\n\n", style="white" if changed else "grey42")
    return text


## ── Result cache entry ──────────────────────────────────────────────────────


@dataclass
class RunState:
    status: str  # "idle" | "running" | "done" | "error"
    row: dict | None = None
    log: str = ""
    error: str = ""


## ── App ─────────────────────────────────────────────────────────────────────


class PACERAblationTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #sidebar { width: 38; border: solid $accent; }
    #main { width: 1fr; }
    #breadcrumb { height: 3; content-align: left middle; padding: 0 1; background: $boost; }
    #panes { height: 1fr; }
    #config-pane { width: 1fr; border: solid $secondary; padding: 0 1; }
    #results-pane { width: 1fr; border: solid $secondary; padding: 0 1; }
    #status-line { height: 1; padding: 0 1; color: $text-muted; }
    ListView { height: 1fr; }
    """

    BINDINGS = [
        ("left,h", "prev", "Prev experiment"),
        ("right,l", "next", "Next experiment"),
        ("r", "run_current", "Run current"),
        ("a", "run_all_remaining", "Run rest"),
        ("q", "quit", "Quit"),
    ]

    index: reactive[int] = reactive(0)

    def __init__(self, lasa_pattern: str = "GShape") -> None:
        super().__init__()
        self.lasa_pattern = lasa_pattern
        self.items = flatten_items()
        self.states: dict[int, RunState] = {
            i: RunState(status="idle") for i in range(len(self.items))
        }
        self.baseline_cfg = make_baseline_pacer_config()
        self.demonstrations = None
        self.bc_policy_loss: float | None = None
        self.ready = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield LoadingIndicator(id="loading")
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield ListView(id="nav-list")
            with Vertical(id="main"):
                yield Static(id="breadcrumb")
                with Horizontal(id="panes"):
                    yield VerticalScroll(Static(id="config-text"), id="config-pane")
                    yield VerticalScroll(
                        DataTable(id="results-table"), id="results-pane"
                    )
                yield Static(id="status-line")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#body").display = False
        for item in self.items:
            self.query_one("#nav-list", ListView).append(
                ListItem(Static(self._nav_label(item, "idle")))
            )
        table = self.query_one("#results-table", DataTable)
        table.add_columns("metric", "value")
        self.load_baseline()

    def _nav_label(self, item: Item, status: str) -> str:
        icon = {
            "idle": "\u25cb",
            "running": "\u23f3",
            "done": "\u2713",
            "error": "\u2717",
        }[status]
        return f"{icon} {item.group} / {item.variant}"

    ## ── Startup ──────────────────────────────────────────────────────────────

    @work(thread=True)
    def load_baseline(self) -> None:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                demonstrations = DemonstrationLoader(
                    config=DemonstrationLoaderConfig(
                        choice="FROM_LASA",
                        LASA_pattern=self.lasa_pattern,
                        filepath=None,
                        corruptions_choice=None,
                    )
                ).load()
                bc_result = BCExperiment(
                    demonstrations, bc_train_config=BASELINE_BC_TRAIN_CONFIG
                ).run()
            self.call_from_thread(
                self._on_baseline_loaded,
                demonstrations,
                float(bc_result.bc_policy_loss.detach()),
            )
        except Exception:
            self.call_from_thread(self._on_baseline_failed, traceback.format_exc())

    def _on_baseline_loaded(self, demonstrations, bc_policy_loss: float) -> None:
        self.demonstrations = demonstrations
        self.bc_policy_loss = bc_policy_loss
        self.ready = True
        self.query_one("#loading").display = False
        self.query_one("#body").display = True
        self.refresh_view()
        self.notify(f"Ready. BC-only baseline loss = {bc_policy_loss:.4f}")

    def _on_baseline_failed(self, tb: str) -> None:
        self.query_one("#loading", LoadingIndicator).display = False
        self.query_one("#status-line", Static).update(
            f"[red]Failed to load baseline:[/red]\n{tb}"
        )

    ## ── Navigation ──────────────────────────────────────────────────────────

    def action_prev(self) -> None:
        if self.ready:
            self.index = max(0, self.index - 1)

    def action_next(self) -> None:
        if self.ready:
            self.index = min(len(self.items) - 1, self.index + 1)

    def watch_index(self, index: int) -> None:
        if not self.ready:
            return
        nav = self.query_one("#nav-list", ListView)
        nav.index = index
        self.refresh_view()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is not None:
            self.index = idx

    ## ── Rendering the current experiment ───────────────────────────────────

    def refresh_view(self) -> None:
        item = self.items[self.index]
        state = self.states[self.index]

        self.query_one("#breadcrumb", Static).update(
            f"[b]{item.group}[/b] / [b]{item.variant}[/b]   "
            f"({self.index + 1}/{len(self.items)})\n[dim]{item.description}[/dim]"
        )

        variant_cfg = item.mutate(make_baseline_pacer_config())
        self.query_one("#config-text", Static).update(
            describe_config(self.baseline_cfg, variant_cfg)
        )

        self._render_results(state)
        self._update_status(state)

    def _render_results(self, state: RunState) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear()
        if state.status == "done" and state.row:
            for key, value in state.row.items():
                if key in ("group", "variant", "description"):
                    continue
                shown = f"{value:.5f}" if isinstance(value, float) else str(value)
                table.add_row(key, shown)
        elif state.status == "error":
            table.add_row(
                "error",
                state.error.splitlines()[-1] if state.error else "unknown error",
            )

    def _update_status(self, state: RunState) -> None:
        line = self.query_one("#status-line", Static)
        hint = "press 'r' to run this experiment, 'a' to run it and the rest"
        if state.status == "idle":
            line.update(f"[grey58]not run yet -- {hint}[/grey58]")
        elif state.status == "running":
            line.update("[yellow]running...[/yellow]")
        elif state.status == "done":
            line.update("[green]done[/green]")
        elif state.status == "error":
            line.update("[red]run failed -- see results pane / notifications[/red]")

    ## ── Running experiments ─────────────────────────────────────────────────

    def action_run_current(self) -> None:
        self._launch_run(self.index)

    def action_run_all_remaining(self) -> None:
        pending = [
            i
            for i in range(self.index, len(self.items))
            if self.states[i].status != "done"
        ]
        if pending:
            self._launch_run_many(pending)

    def _mark(self, idx: int, status: str) -> None:
        self.states[idx].status = status
        try:
            nav = self.query_one("#nav-list", ListView)
            label = nav.children[idx].query_one(Static)
            label.update(self._nav_label(self.items[idx], status))
        except Exception:
            pass
        if idx == self.index:
            self._update_status(self.states[idx])

    def _launch_run(self, idx: int) -> None:
        if not self.ready or self.states[idx].status == "running":
            return
        self._mark(idx, "running")
        self._run_worker([idx])

    def _launch_run_many(self, indices: list[int]) -> None:
        if not self.ready:
            return
        for idx in indices:
            if self.states[idx].status != "running":
                self._mark(idx, "running")
        self._run_worker(indices)

    @work(thread=True, exclusive=True)
    def _run_worker(self, indices: list[int]) -> None:
        for idx in indices:
            item = self.items[idx]
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    cfg = item.mutate(make_baseline_pacer_config())
                    pacer_bc_result = PACERBCExperiment(
                        self.demonstrations,
                        pacer_config=cfg,
                        bc_train_config=BASELINE_BC_TRAIN_CONFIG,
                    ).run()
                    row = collect_row(
                        group=item.group,
                        variant=item.variant,
                        description=item.description,
                        demonstrations=self.demonstrations,
                        bc_policy_loss=self.bc_policy_loss,
                        pacer_bc_result=pacer_bc_result,
                    )
                self.call_from_thread(self._on_run_done, idx, row, buf.getvalue())
            except Exception:
                self.call_from_thread(
                    self._on_run_failed, idx, traceback.format_exc(), buf.getvalue()
                )

    def _on_run_done(self, idx: int, row: dict, log: str) -> None:
        self.states[idx] = RunState(status="done", row=row, log=log)
        self._mark(idx, "done")
        if idx == self.index:
            self.refresh_view()

    def _on_run_failed(self, idx: int, tb: str, log: str) -> None:
        self.states[idx] = RunState(status="error", error=tb, log=log)
        self._mark(idx, "error")
        if idx == self.index:
            self.refresh_view()
        self.notify(
            f"{self.items[idx].group}/{self.items[idx].variant} failed",
            severity="error",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lasa-pattern", default="GShape")
    args = parser.parse_args()
    PACERAblationTUI(lasa_pattern=args.lasa_pattern).run()
