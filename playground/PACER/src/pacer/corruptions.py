"""
Dataset corruptions
=======
"""
# src/pacer/corruptions.py

## ── Imports ──────────────────────────────────────────────────────────────────

import random
from dataclasses import dataclass
from typing import Generic

import numpy as np
import numpy.linalg as la
from typingkit.core import TypedList

from pacer.base import Demonstration, Demonstrations
from pacer.typings import Action, DimAction, DimState, NumDemos, NumPoints
from pacer.utils import EPS, SEED, set_seed

## ── Corruptions ──────────────────────────────────────────────────────────────


@dataclass
class DemonstrationCorrupter(Generic[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    noise_std: float = 0.0
    outlier_fraction: float = 0.0
    outlier_scale: float = 3.0
    bias_strength: float = 0.0
    dropout_fraction: float = 0.0
    seed: int = SEED

    def inject_corruptions(
        self,
    ) -> Demonstrations[NumDemos, NumPoints, DimState, DimAction]:
        set_seed(self.seed)
        corrupted_demos = TypedList[
            NumDemos, Demonstration[NumPoints, DimState, DimAction]
        ]()
        for demo in self.demonstrations:
            new_actions = TypedList[NumPoints, Action[DimAction]]()
            bias_vector = np.random.randn(demo.action_dim)
            bias_vector /= la.norm(bias_vector) + EPS
            for action in demo.actions:
                a = action.copy()

                # Gaussian noise
                if self.noise_std > 0:
                    a += np.random.normal(0, self.noise_std, size=a.shape)

                # Outliers
                if (
                    self.outlier_fraction > 0
                    and random.random() < self.outlier_fraction
                ):
                    a += self.outlier_scale * np.random.randn(*a.shape)

                # Systematic bias
                if self.bias_strength > 0:
                    a += self.bias_strength * bias_vector

                # Dropout
                if (
                    self.dropout_fraction > 0
                    and random.random() < self.dropout_fraction
                ):
                    a = np.zeros_like(a)

                new_actions.append(Action[DimAction](a))
            corrupted_demos.append(
                Demonstration(
                    index=demo.index, states=demo.states.copy(), actions=new_actions
                )
            )
        return Demonstrations(demos=corrupted_demos)


## ─────────────────────────────────────────────────────────────────────────────
