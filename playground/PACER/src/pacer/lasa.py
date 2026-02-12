"""
LASA Dataset
=======
"""
# src/pacer/lasa.py

from typing import Literal

import numpy as np
import pyLasaDataset as lasa  # type: ignore
from pyLasaDataset.dataset import _Data  # type: ignore
from typed_numpy._typed.helpers import TWO

from .base import Array3D, Demonstration, Demonstrations, npDType

SEVEN = Literal[7]
THOUSAND = Literal[1000]


class LASADemonstrations:
    def __init__(self, data: _Data = lasa.DataSet.GShape) -> None:
        self.data = data

        Ar7k2 = Array3D[SEVEN, THOUSAND, TWO]
        self.positions = Ar7k2([demo.__getattribute__("pos").T for demo in data.demos])
        self.velocities = Ar7k2([demo.__getattribute__("vel").T for demo in data.demos])
        self.positions_diff = np.diff(
            self.positions, axis=-2, append=np.zeros((7, 1, 2), dtype=npDType)
        )

    def to_demonstrations(self) -> Demonstrations[TWO, TWO]:
        return Demonstrations(
            [
                Demonstration(
                    index=index,  # i
                    states=[state for state in states],
                    actions=[action for action in actions],
                )
                for index, (states, actions) in enumerate(
                    zip(self.positions, self.velocities)
                )
            ]
        )
