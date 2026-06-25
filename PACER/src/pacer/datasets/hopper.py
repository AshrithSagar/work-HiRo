"""
Hopper
=======
"""
# src/pacer/datasets/hopper.py

## ── Imports ──────────────────────────────────────────────────────────────────

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
    def _files(self) -> TypedList[SIXTEEN, Path]:
        return TypedList[SIXTEEN, Path](sorted(self.path.glob("*.npz")))

    def _load_demo(
        self, demo_index: DemoIndex, file: Path
    ) -> Demonstration[Any, ELEVEN, THREE]:
        data = np.load(file, allow_pickle=True)
        return Demonstration(
            index=demo_index,
            states=States[Any, ELEVEN](
                State[ELEVEN](state) for state in data["states"]
            ),
            actions=Actions[Any, THREE](
                Action[THREE](action) for action in data["actions_exec"]
            ),
        )

    def _preview_value(self, arr: np.ndarray, max_items: int = 5) -> str:
        if np.ndim(arr) == 0:
            return repr(arr.item())
        flat = arr.ravel()
        if flat.size <= max_items:
            return np.array2string(flat, precision=3)
        head = np.array2string(flat[:max_items], precision=3)
        return f"{head[:-1]}, ...]"

    def preview(self) -> None:
        for file in self._files:
            data = np.load(file, allow_pickle=True)

            table = Table(title=file.name, show_header=True)
            table.add_column("key", style="magenta")
            table.add_column("shape", style="green")
            table.add_column("dtype", style="yellow")
            table.add_column("value", style="cyan")
            for key in data.files:
                arr = data[key]
                table.add_row(
                    key, str(arr.shape), str(arr.dtype), self._preview_value(arr)
                )
            console.print(table)
            console.rule()

    def __len__(self) -> SIXTEEN:
        assert self._files.length == 16
        return 16

    def to_demonstrations(self) -> Demonstrations[SIXTEEN, Any, ELEVEN, THREE]:
        return Demonstrations(
            TypedList[SIXTEEN, Demonstration[Any, ELEVEN, THREE]](
                [
                    self._load_demo(demo_index, file)
                    for demo_index, file in enumerate(self._files)
                ]
            )
        )


## ─────────────────────────────────────────────────────────────────────────────
