"""
LASA Dataset
=======
"""
# src/pacer/lasa.py

# pyright: reportPrivateUsage = false

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
import pyLasaDataset as lasa  # type: ignore[import-untyped]  # ty: ignore[unused-ignore-comment]
from pyLasaDataset.dataset import (  # type: ignore[import-untyped]  # ty: ignore[unused-ignore-comment]
    _Data,
)
from typingkit.core import TypedList
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
    def __init__(self, data: _Data = lasa.DataSet.GShape) -> None:
        self.data = data

        self.positions = Array_7x1000x2(
            [demo.__getattribute__("pos").T for demo in data.demos], dtype=npDType
        )
        self.velocities = Array_7x1000x2(
            [demo.__getattribute__("vel").T for demo in data.demos], dtype=npDType
        )
        self.positions_diff = np.diff(
            self.positions, axis=-2, append=np.zeros((7, 1, 2), dtype=npDType)
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


class LASADataSet3D(Generic[_ScalarT]):
    def __init__(
        self,
        data: _Data = lasa.DataSet.GShape,
        *,
        dtype: type[_ScalarT] = npDType,  # type: ignore[assignment]
    ) -> None:
        dataset = LASADataSet(data)

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
