"""
Hopper
=======
"""
# src/pacer/datasets/hopper.py

## ── Imports ──────────────────────────────────────────────────────────────────

import os
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from rich.table import Table
from typingkit.core import TypedList
from typingkit.numpy._typed.helpers import THREE

from pacer import console
from pacer.base import Action, Actions, Demonstration, Demonstrations, State, States
from pacer.typings import DemoIndex

## ── Typings ──────────────────────────────────────────────────────────────────

ELEVEN: TypeAlias = Literal[11]
SIXTEEN: TypeAlias = Literal[16]

## ── Hopper ───────────────────────────────────────────────────────────────────


class HopperDataset:
    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = Path(__file__).resolve().parents[4] / "repos/PACER/datasets/hopper"
        self.path: Path = Path(path)

    @cached_property
    def filenames(self) -> TypedList[SIXTEEN, str]:
        return TypedList[SIXTEEN, str](sorted(os.listdir(self.path)))

    def _load_demo(
        self, demo_index: DemoIndex, filename: str
    ) -> Demonstration[Any, ELEVEN, THREE]:
        data = np.load(self.path / filename, allow_pickle=True)
        return Demonstration(
            index=demo_index,
            states=States[Any, ELEVEN](
                State[ELEVEN](state) for state in data["states"]
            ),
            actions=Actions[Any, THREE](
                Action[THREE](action) for action in data["actions_exec"]
            ),
        )

    def preview(self) -> None:
        for filename in self.filenames:
            data = np.load(self.path / filename, allow_pickle=True)

            table = Table(title=filename, show_header=True)
            table.add_column("key", style="magenta")
            table.add_column("shape", style="green")
            table.add_column("dtype", style="yellow")
            table.add_column("value", style="cyan")
            for key in data.keys():
                arr = data[key]
                shape = str(arr.shape) if hasattr(arr, "shape") else "-"
                dtype = str(arr.dtype) if hasattr(arr, "dtype") else "-"
                if np.ndim(arr) == 0:
                    preview = repr(arr.item())
                else:
                    preview = " ..."
                table.add_row(key, shape, dtype, preview)
            console.print(table)
            console.rule()

    def __len__(self) -> SIXTEEN:
        assert len(self.filenames) == 16
        return 16

    def to_demonstrations(self) -> Demonstrations[SIXTEEN, Any, ELEVEN, THREE]:
        return Demonstrations(
            TypedList[SIXTEEN, Demonstration[Any, ELEVEN, THREE]](
                [
                    self._load_demo(demo_index, filename)
                    for demo_index, filename in enumerate(self.filenames)
                ]
            )
        )


## ─────────────────────────────────────────────────────────────────────────────
