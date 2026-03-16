"""
LASA Dataset
=======
"""
# src/pacer/lasa.py

# pyright: reportPrivateUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Literal, TypeAlias, TypeVar

import numpy as np
from pyLASAHandwritingDataset import DataSet, SinglePatternMotion
from typingkit.core import RuntimeGeneric, TypedList
from typingkit.numpy import enforce_shapes
from typingkit.numpy._typed.helpers import THREE, TWO, Array3D

from pacer.base import Demonstration, Demonstrations
from pacer.typings import Actions, States, npDType

## ── Typings ──────────────────────────────────────────────────────────────────

SEVEN: TypeAlias = Literal[7]
THOUSAND: TypeAlias = Literal[1000]

_ScalarT = TypeVar("_ScalarT", bound=np.generic, default=npDType)

Array_7x1000x2: TypeAlias = Array3D[SEVEN, THOUSAND, TWO, np.dtype[npDType]]
Array_7x1000x3: TypeAlias = Array3D[SEVEN, THOUSAND, THREE, np.dtype[_ScalarT]]

## ── LASA ─────────────────────────────────────────────────────────────────────


class LASADataSet:
    def __init__(self, pattern: SinglePatternMotion) -> None:
        self.data = DataSet[pattern]

        self.positions = Array_7x1000x2(
            [demo.pos.T for demo in self.data.demos], dtype=npDType
        )
        self.velocities = Array_7x1000x2(
            [demo.vel.T for demo in self.data.demos], dtype=npDType
        )
        self.positions_diff = Array_7x1000x2(
            np.diff(self.positions, axis=-2, append=np.zeros((7, 1, 2), dtype=npDType))
            / self.data.dt
        )

    def __len__(self) -> SEVEN:
        return 7

    def to_demonstrations(self) -> Demonstrations[SEVEN, THOUSAND, TWO, TWO]:
        return Demonstrations(
            TypedList[SEVEN, Demonstration[THOUSAND, TWO, TWO]](
                [
                    Demonstration(
                        index=index,  # i
                        states=States[THOUSAND, TWO](states),
                        actions=Actions[THOUSAND, TWO](actions),
                    )
                    for index, (states, actions) in enumerate(
                        zip(self.positions, self.velocities)
                    )
                ]
            )
        )


class LASADataSet3D(RuntimeGeneric[_ScalarT]):
    def __init__(
        self,
        pattern: SinglePatternMotion,
        *,
        dtype: type[_ScalarT] = npDType,  # type: ignore[assignment]
    ) -> None:
        dataset = LASADataSet(pattern)

        @enforce_shapes
        def pad(arr: Array_7x1000x2) -> Array_7x1000x3[_ScalarT]:
            out = Array_7x1000x3(np.zeros((7, 1000, 3), dtype=dtype))
            out[:, :, :2] = arr
            return out

        self.positions = pad(dataset.positions)
        self.velocities = pad(dataset.velocities)
        self.positions_diff = pad(dataset.positions_diff)

    def __len__(self) -> SEVEN:
        return 7


## ─────────────────────────────────────────────────────────────────────────────
